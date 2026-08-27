// C++ include files
#include <condition_variable>
#include <memory>
#include <mutex>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

// ROOT headers
#include <TBufferFile.h>
#include <TClass.h>

// CMSSW include files
#include "DataFormats/Common/interface/PathStateToken.h"
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/ProductNamePattern.h"
#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/Concurrency/interface/chain_first.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/GenericHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/WrapperBaseHandle.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/MPICore/interface/MPIChannel.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"
#include "HeterogeneousCore/MPICore/interface/MutableOnceFlag.h"
#include "HeterogeneousCore/MPIServices/interface/MPIConsistencyChecker.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserFactory.h"

class MPISender : public edm::stream::EDProducer<edm::ExternalWork, edm::GlobalCache<MutableOnceFlag>> {
public:
  MPISender(edm::ParameterSet const& config, MutableOnceFlag const* cache)
      : upstream_(consumes<MPIToken>(config.getParameter<edm::InputTag>("upstream"))),
        token_(produces<MPIToken>()),
        instance_(config.getParameter<int32_t>("instance")),
        buffer_(std::make_unique<TBufferFile>(TBuffer::kWrite)),
        metadata_size_(0),
        activity_(config.getParameter<edm::InputTag>("activity")),
        enableTrivialSerialisation_(config.getUntrackedParameter<bool>("enableTrivialSerialisation")) {
    // instance 0 is reserved for the MPIController / MPISource pair
    // instance values greater than 255 may not fit in the MPI tag
    if (instance_ < 1 or instance_ > 255) {
      throw cms::Exception("InvalidValue") << "Invalid MPISender instance value, please use a value between 1 and 255";
    }

    if (not activity_.label().empty()) {
      activityToken_ = consumes<edm::PathStateToken>(activity_);
    }

    auto const& products = config.getParameter<std::vector<edm::ParameterSet>>("products");
    products_.reserve(products.size());
    for (auto const& product : products) {
      auto const& type = product.getParameter<std::string>("type");
      auto const& input_tag = product.getParameter<edm::InputTag>("name");
      Entry entry;
      entry.type = edm::TypeWithDict::byName(type);
      entry.wrappedType = edm::TypeWithDict::byName("edm::Wrapper<" + type + ">");
      entry.token = this->consumes(edm::TypeToGet{edm::TypeID{entry.type.typeInfo()}, edm::PRODUCT_TYPE}, input_tag);

      LogDebug("MPISender") << "send product \"" << input_tag.label() << '_' << input_tag.instance() << '_'
                            << input_tag.process() << "\" of type \"" << entry.type.name()
                            << "\" over MPI channel instance " << instance_;

      products_.emplace_back(std::move(entry));
    }

    // record information about this sender for configuration consistency check
    edm::Service<MPIConsistencyChecker> module_info_service;
    std::vector<std::string> product_types;
    product_types.reserve(products_.size());
    for (auto const& entry : products_) {
      product_types.push_back(entry.type.name());
    }
    std::string module_label = config.getParameter<std::string>("@module_label");
    std::string upstream_label = config.getParameter<edm::InputTag>("upstream").label();
    if (cache == nullptr) {
      throw cms::Exception("MPISender") << "MPISender's global cache is null";
    }
    std::call_once(cache->information_recorded_flag, [&]() {
      module_info_service->recordMPIModuleInfo(true, module_label, upstream_label, this->instance_, product_types);
    });
  }

  static std::unique_ptr<MutableOnceFlag> initializeGlobalCache(edm::ParameterSet const&) {
    return std::make_unique<MutableOnceFlag>();
  }

  static void globalEndJob(MutableOnceFlag const*) {}

  void acquire(edm::Event const& event, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder holder) final {
    const MPIToken& token = event.get(upstream_);
    // pass the number of products to estimate the right size for the metadata buffer
    auto meta = std::make_shared<ProductMetadataBuilder>(products_.size());

    // We use std::shared_ptr, instead of std::unique_ptr, so that readers can
    // be captured by move by runAsync's lamnda. This is ultimately because this
    // lambda is used to construct an std::function, which requires its callable
    // to be copy-constructible.
    std::vector<std::shared_ptr<const ngt::ReaderBase>> readers;
    readers.reserve(products_.size());
    size_t index = 0;
    buffer_->Reset();
    has_serialized_ = false;
    is_active_ = true;

    if (not activity_.label().empty()) {
      const edm::Handle<edm::PathStateToken>& pathStateTokenHandle = event.getHandle(activityToken_);

      if (!pathStateTokenHandle.isValid()) {
        meta->setProductCount(-1);
        is_active_ = false;
      }
    }

    if (is_active_) {
      for (auto const& entry : products_) {
        // Get the product
        edm::Handle<edm::WrapperBase> handle(entry.type.typeInfo());
        event.getByToken(entry.token, handle);

        if (handle.isValid()) {
          edm::WrapperBase const* wrapper = handle.product();
          std::unique_ptr<ngt::SerialiserBase> serialiser;
          if (enableTrivialSerialisation_) {
            serialiser = ngt::SerialiserFactory::get()->tryToCreate(entry.type.typeInfo().name());
          }

          if (serialiser) {
            LogDebug("MPISender") << "Found serializer for type \"" << entry.type.name() << "\" ("
                                  << entry.type.typeInfo().name() << ")";
            auto reader = serialiser->reader(*wrapper);
            ngt::AnyBuffer buffer = reader->parameters();
            meta->addTrivialCopy(buffer.data(), buffer.size_bytes());
            readers.push_back(std::move(reader));
          } else {
            LogDebug("MPISender") << "No serializer for type \"" << entry.type.name() << "\" ("
                                  << entry.type.typeInfo().name() << "), using ROOT serialization";
            TClass* cls = entry.wrappedType.getClass();
            if (!cls) {
              throw cms::Exception("MPISender") << "Failed to get TClass for type: " << entry.type.name();
            }
            size_t bufLen = serializeAndStoreBuffer_(index, cls, wrapper);
            meta->addSerialized(bufLen);
            has_serialized_ = true;
          }

        } else {
          // handle missing product
          meta->addMissing();
        }
        index++;
      }
    }

    // Submit sending of all products to run in the additional asynchronous threadpool
    edm::Service<edm::Async> as;
    as->runAsync(
        std::move(holder),
        [this, token, meta = std::move(meta), readers = std::move(readers)]() {
          token.channel()->sendMetadata(instance_, meta);
          if (has_serialized_) {
#ifdef EDM_ML_DEBUG
            {
              edm::LogSystem msg("MPISender");
              msg << "Sending serialised product:\n";
              for (int i = 0; i < buffer_->Length(); ++i) {
                msg << "0x" << std::hex << std::setw(2) << std::setfill('0')
                    << (unsigned int)(unsigned char)buffer_->Buffer()[i] << (i % 16 == 15 ? '\n' : ' ');
              }
            }
#endif
            token.channel()->sendBuffer(buffer_->Buffer(), buffer_->Length(), instance_, EDM_MPI_SendSerializedProduct);
          }
          for (auto const& reader : readers) {
            token.channel()->sendTrivialCopyProduct(instance_, *reader);
          }
        },
        []() { return "Calling MPISender::acquire()"; });
  }

  void produce(edm::Event& event, edm::EventSetup const&) final {
    // write a shallow copy of the channel to the output, so other modules can consume it
    // to indicate that they should run after this
    MPIToken token = event.get(upstream_);
    event.emplace(token_, token);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    descriptions.setComment(
        "This module can consume arbitrary event products and copy them to an \"MPIReceiver\" module in a separate "
        "CMSSW job.");

    edm::ParameterSetDescription product;
    product.add<std::string>("type")->setComment("C++ type of the product to be sent.");
    product.add<edm::InputTag>("name")->setComment("Input tag of the product to be sent.");

    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("upstream", {"source"})
        ->setComment(
            "MPI communication channel. Can be an \"MPIController\", \"MPISource\", \"MPISender\" or \"MPIReceiver\". "
            "Passing an \"MPIController\" or \"MPISource\" only identifies the pair of local and remote application "
            "that communicate. Passing an \"MPISender\" or \"MPIReceiver\" in addition imposes a scheduling "
            "dependency.");
    desc.addVPSet("products", product, {})
        ->setComment(
            "Event products to be consumed and copied over to a separate CMSSW job."
            "Is configured as a vector of parameter sets, each containing the C++ type and input tag of a product.");
    desc.add<int32_t>("instance", 0)
        ->setComment("A value between 1 and 255 used to identify a matching pair of \"MPISender\"/\"MPIReceiver\".");
    desc.add<edm::InputTag>("activity", edm::InputTag(""))
        ->setComment(
            "Activity product. If empty (default), sender is always active. "
            "If set but missing in event, the sender skips transfer.");
    desc.addUntracked<bool>("enableTrivialSerialisation", true)
        ->setComment(
            "If true (default), use the trivial serialisation mechanism for supported types. If false, use "
            "ROOT serialisation for all types. Intended to be disabled only for benchmarking purposes");

    descriptions.addWithDefaultLabel(desc);
  }

private:
  size_t serializeAndStoreBuffer_(size_t index, TClass* type, void const* product) {
    size_t size = buffer_->Length();
    type->Streamer(const_cast<void*>(product), *buffer_);
    return buffer_->Length() - size;
  }

  struct Entry {
    edm::TypeWithDict type;
    edm::TypeWithDict wrappedType;
    edm::EDGetToken token;
  };

  edm::EDGetTokenT<MPIToken> const upstream_;  // MPIToken used to establish the communication channel
  edm::EDPutTokenT<MPIToken> const token_;  // copy of the MPIToken that may be used to implement an ordering relation
  std::vector<Entry> products_;             // types and tokens corresponding to the branches
  int32_t const instance_;                  // instance used to identify the source-destination pair
  std::unique_ptr<TBufferFile> buffer_;
  size_t metadata_size_;
  edm::InputTag activity_;
  edm::EDGetTokenT<edm::PathStateToken> activityToken_;
  bool is_active_ = true;
  bool enableTrivialSerialisation_ = true;
  bool has_serialized_ = false;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MPISender);
