// C++ include files
#include <condition_variable>
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
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserFactory.h"

class MPISender : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  MPISender(edm::ParameterSet const& config)
      : upstream_(consumes<MPIToken>(config.getParameter<edm::InputTag>("upstream"))),
        token_(produces<MPIToken>()),
        patterns_(edm::productPatterns(config.getParameter<std::vector<std::string>>("products"))),
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

    products_.resize(patterns_.size());

    if (not activity_.label().empty()) {
      activityToken_ = consumes<edm::PathStateToken>(activity_);
    }

    callWhenNewProductsRegistered([this](edm::ProductDescription const& product) {
      static const std::string_view kPathStatus("edm::PathStatus");
      static const std::string_view kEndPathStatus("edm::EndPathStatus");

      // std::cout << "MPISender: considering product " << product.friendlyClassName() << '_'
      //           << product.moduleLabel() << '_' << product.productInstanceName() << '_' << product.processName()
      //           << " of type " << product.unwrappedType().name() << " branch type " << product.branchType() << "\n";

      switch (product.branchType()) {
        case edm::InEvent:
          if (product.className() == kPathStatus or product.className() == kEndPathStatus)
            return;
          for (size_t pattern_index = 0; pattern_index < patterns_.size(); pattern_index++) {
            if (patterns_[pattern_index].match(product)) {
              Entry entry;
              entry.type = product.unwrappedType();
              entry.wrappedType = product.wrappedType();
              // TODO move this to EDConsumerBase::consumes() ?
              entry.token = this->consumes(
                  edm::TypeToGet{product.unwrappedTypeID(), edm::PRODUCT_TYPE},
                  edm::InputTag{product.moduleLabel(), product.productInstanceName(), product.processName()});

              LogDebug("MPISender") << "send product \"" << product.friendlyClassName() << '_' << product.moduleLabel()
                                    << '_' << product.productInstanceName() << '_' << product.processName()
                                    << "\" of type \"" << entry.type.name() << "\" over MPI channel instance "
                                    << instance_;

              products_[pattern_index] = std::move(entry);
              break;
            }
          }
          break;

        case edm::InLumi:
        case edm::InRun:
        case edm::InProcess:
          // lumi, run and process products are not supported
          break;

        default:
          throw edm::Exception(edm::errors::LogicError)
              << "Unexpected branch type " << product.branchType() << "\nPlease contact a Framework developer\n";
      }
    });

    // TODO add an error if a pattern does not match any branches? how?
  }

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

    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("upstream", {"source"})
        ->setComment(
            "MPI communication channel. Can be an \"MPIController\", \"MPISource\", \"MPISender\" or \"MPIReceiver\". "
            "Passing an \"MPIController\" or \"MPISource\" only identifies the pair of local and remote application "
            "that communicate. Passing an \"MPISender\" or \"MPIReceiver\" in addition imposes a scheduling "
            "dependency.");
    desc.add<std::vector<std::string>>("products", {})
        ->setComment(
            "Event products to be consumed and copied over to a separate CMSSW job. Can be a list of module labels, "
            "branch names (similar to an OutputModule's \"keep ...\" statement), or a mix of the two. Wildcards (\"?\" "
            "and \"*\") are allowed in a module label or in each field of a branch name.");
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
  std::vector<edm::ProductNamePattern> patterns_;  // branches to read from the Event and send over the MPI channel
  std::vector<Entry> products_;                    // types and tokens corresponding to the branches
  int32_t const instance_;                         // instance used to identify the source-destination pair
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
