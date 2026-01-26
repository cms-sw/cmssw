// C++ include files
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <utility>

// ROOT headers
#include <TBufferFile.h>
#include <TClass.h>

// CMSSW include files
#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/Concurrency/interface/chain_first.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/WrapperBaseOrphanHandle.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/MPICore/interface/api.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserFactory.h"

class MPIReceiver : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  MPIReceiver(edm::ParameterSet const& config)
      : upstream_(consumes<MPIToken>(config.getParameter<edm::InputTag>("upstream"))),
        token_(produces<MPIToken>()),
        instance_(config.getParameter<int32_t>("instance"))  //
  {
    // instance 0 is reserved for the MPIController / MPISource pair
    // instance values greater than 255 may not fit in the MPI tag
    if (instance_ < 1 or instance_ > 255) {
      throw cms::Exception("InvalidValue")
          << "Invalid MPIReceiver instance value, please use a value between 1 and 255";
    }

    auto const& products = config.getParameter<std::vector<edm::ParameterSet>>("products");
    products_.reserve(products.size());
    for (auto const& product : products) {
      auto const& type = product.getParameter<std::string>("type");
      auto const& label = product.getParameter<std::string>("label");
      Entry entry;
      entry.type = edm::TypeWithDict::byName(type);
      entry.wrappedType = edm::TypeWithDict::byName("edm::Wrapper<" + type + ">");
      entry.token = produces(edm::TypeID{entry.type.typeInfo()}, label);

      LogTrace("MPIReceiver") << "receive type \"" << entry.type.name() << "\" for label \"" << label
                              << "\" over MPI channel instance " << this->instance_;

      products_.emplace_back(std::move(entry));
    }
  }

  void acquire(edm::Event const& event, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder holder) final {
    const MPIToken& token = event.get(upstream_);

    // also try unique or optional
    received_meta_ = std::make_shared<ProductMetadataBuilder>();

    edm::Service<edm::Async> as;
    as->runAsync(
        std::move(holder),
        [this, token]() { token.channel()->receiveMetadata(instance_, received_meta_); },
        []() { return "Calling MPIReceiver::acquire()"; });
  }

  void produce(edm::Event& event, edm::EventSetup const&) final {
    // read the MPIToken used to establish the communication channel
    MPIToken token = event.get(upstream_);
#ifdef EDM_ML_DEBUG
    // dump the summary of metadata
    received_meta_->debugPrintMetadataSummary();
#endif

    // if filter was false before the sender, receive nothing
    if (received_meta_->productCount() == -1) {
      event.emplace(token_, token);
      return;
    }

    std::unique_ptr<TBufferFile> serialized_buffer;
    if (received_meta_->hasSerialized()) {
      serialized_buffer = token.channel()->receiveSerializedBuffer(instance_, received_meta_->serializedBufferSize());
#ifdef EDM_ML_DEBUG
      {
        edm::LogSystem msg("MPISender");
        msg << "Received serialised product:\n";
        for (int i = 0; i < received_meta_->serializedBufferSize(); ++i) {
          msg << "0x" << std::hex << std::setw(2) << std::setfill('0')
              << (unsigned int)(unsigned char)serialized_buffer->Buffer()[i] << (i % 16 == 15 ? '\n' : ' ');
        }
      }
#endif
    }

    for (auto const& entry : products_) {
      auto product_meta = received_meta_->getNext();
      if (product_meta.kind == ProductMetadata::Kind::Missing) {
        edm::LogWarning("MPIReceiver") << "Product " << entry.type.name() << " was not received.";
        continue;  // Skip products that weren't sent
      }

      else if (product_meta.kind == ProductMetadata::Kind::Serialized) {
        std::unique_ptr<edm::WrapperBase> wrapper(
            reinterpret_cast<edm::WrapperBase*>(entry.wrappedType.getClass()->New()));
        assert(static_cast<int32_t>(serialized_buffer->Length() + product_meta.sizeMeta) <=
                   received_meta_->serializedBufferSize() &&
               "serialized data buffer is shorter than expected");
        entry.wrappedType.getClass()->Streamer(wrapper.get(), *serialized_buffer);
        // put the data into the Event
        event.put(entry.token, std::move(wrapper));
      }

      else if (product_meta.kind == ProductMetadata::Kind::TrivialCopy) {
        std::unique_ptr<ngt::SerialiserBase> serialiser =
            ngt::SerialiserFactory::get()->tryToCreate(entry.type.typeInfo().name());
        if (not serialiser) {
          throw cms::Exception("SerializerError") << "Receiver could not retrieve its serializer when it was expected";
        }
        auto writer = serialiser->writer();
        ngt::AnyBuffer buffer = writer->uninitialized_parameters();  // constructs buffer with typeid
        assert(buffer.size_bytes() == product_meta.sizeMeta);
        std::memcpy(buffer.data(), product_meta.trivialCopyOffset, product_meta.sizeMeta);
        writer->initialize(buffer);
        token.channel()->receiveInitializedTrivialCopy(instance_, *writer);
        writer->finalize();
        // put the data into the Event
        event.put(entry.token, writer->get());
      }
    }

    if (received_meta_->hasSerialized()) {
      assert(serialized_buffer->Length() == received_meta_->serializedBufferSize() &&
             "serialized data buffer is not equal to the expected length");
    }

    // write a shallow copy of the channel to the output, so other modules can consume it
    // to indicate that they should run after this
    event.emplace(token_, token);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    descriptions.setComment(
        "This module can receive arbitrary event products from an \"MPISender\" module in a separate CMSSW job, and "
        "produce them into the event.");

    edm::ParameterSetDescription product;
    product.add<std::string>("type")->setComment("C++ type of the product to be received.");
    product.add<std::string>("label", "")->setComment("Product instance label to be associated to the product.");

    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("upstream", {"source"})
        ->setComment(
            "MPI communication channel. Can be an \"MPIController\", \"MPISource\", \"MPISender\" or \"MPIReceiver\". "
            "Passing an \"MPIController\" or \"MPISource\" only identifies the pair of local and remote application "
            "that communicate. Passing an \"MPISender\" or \"MPIReceiver\" in addition imposes a scheduling "
            "dependency.");
    desc.addVPSet("products", product, {})
        ->setComment("Products to be received by a separate CMSSW job and produced into the event.");
    desc.add<int32_t>("instance", 0)
        ->setComment("A value between 1 and 255 used to identify a matching pair of \"MPISender\"/\"MPIReceiver\".");

    descriptions.addWithDefaultLabel(desc);
  }

private:
  struct Entry {
    edm::TypeWithDict type;
    edm::TypeWithDict wrappedType;
    edm::EDPutToken token;
  };

  edm::EDGetTokenT<MPIToken> const upstream_;  // MPIToken used to establish the communication channel
  edm::EDPutTokenT<MPIToken> const token_;  // copy of the MPIToken that may be used to implement an ordering relation
  std::vector<Entry> products_;             // data to be read over the channel and put into the Event
  int32_t const instance_;                  // instance used to identify the source-destination pair

  std::shared_ptr<ProductMetadataBuilder> received_meta_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MPIReceiver);
