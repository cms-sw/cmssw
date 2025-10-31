// C++ include files
#include <utility>

// CMSSW include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/WrapperBaseOrphanHandle.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"

#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/Concurrency/interface/chain_first.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <condition_variable>
#include <mutex>
#include <cassert>

// local include files
#include "api.h"
#include <TBufferFile.h>
#include <TClass.h>

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

      edm::LogVerbatim("MPIReceiver") << "receive type \"" << entry.type.name() << "\" for label \"" << label
                                      << "\" over MPI channel instance " << this->instance_;

      products_.emplace_back(std::move(entry));
    }
  }

  void acquire(edm::Event const& event, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder holder) final {
    MPIToken token = event.get(upstream_);

    //also try unique or optional
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
    // see the summary of metadata for dubug purposes
    // received_meta_->debugPrintMetadataSummary();

    // if filter was false before the sender, receive nothing
    if (received_meta_->productCount() == -1) {
      event.emplace(token_, token);
      return;
    }

    char* buf_ptr = nullptr;
    size_t full_buffer_size = 0;
    size_t buffer_offset_ = 0;

    std::unique_ptr<TBufferFile> serialized_buffer;

    if (received_meta_->hasSerialized()) {
      serialized_buffer = token.channel()->receiveSerializedBuffer(instance_, received_meta_->serializedBufferSize());
      buf_ptr = serialized_buffer->Buffer();
      full_buffer_size = serialized_buffer->BufferSize();
    }

    for (auto const& entry : products_) {
      std::unique_ptr<edm::WrapperBase> wrapper(
          reinterpret_cast<edm::WrapperBase*>(entry.wrappedType.getClass()->New()));

      auto product_meta = received_meta_->getNext();

      if (product_meta.kind == ProductMetadata::Kind::Missing) {
        edm::LogWarning("MPIReceiver") << "Product " << entry.type.name() << " was not received.";
        continue;  // Skip products that weren't sent
      }

      else if (product_meta.kind == ProductMetadata::Kind::Serialized) {
        auto productBuffer = TBufferFile(TBuffer::kRead, product_meta.sizeMeta);
        assert(!wrapper->hasTrivialCopyTraits() && "mismatch between expected and factual metadata type");
        assert(buffer_offset_ < full_buffer_size && "serialized data buffer is shorter than expected");
        productBuffer.SetBuffer(buf_ptr + buffer_offset_, product_meta.sizeMeta, kFALSE /* adopt = false */);
        buffer_offset_ += product_meta.sizeMeta;
        entry.wrappedType.getClass()->Streamer(wrapper.get(), productBuffer);
      }

      else if (product_meta.kind == ProductMetadata::Kind::TrivialCopy) {
        assert(wrapper->hasTrivialCopyTraits() && "mismatch between expected and factual metadata type");
        wrapper->markAsPresent();
        edm::AnyBuffer buffer = wrapper->trivialCopyParameters();  // constructs buffer with typeid
        assert(buffer.size_bytes() == product_meta.sizeMeta);
        // TDL: can we add func to AnyBuffer to replace pointer to the data?
        std::memcpy(buffer.data(), product_meta.trivialCopyOffset, product_meta.sizeMeta);
        wrapper->trivialCopyInitialize(buffer);
        token.channel()->receiveInitializedTrivialCopy(instance_, wrapper.get());
        wrapper->trivialCopyFinalize();
      }
      // put the data into the Event
      event.put(entry.token, std::move(wrapper));
    }

    if (received_meta_->hasSerialized()) {
      assert(static_cast<int>(buffer_offset_) == received_meta_->serializedBufferSize() &&
             "serialized data buffer is not equal to the expected length");
    }

    // write a shallow copy of the channel to the output, so other modules can consume it
    // to indicate that they should run after this
    event.emplace(token_, token);
  }

private:
  struct Entry {
    edm::TypeWithDict type;
    edm::TypeWithDict wrappedType;
    edm::EDPutToken token;
  };

  // TODO consider if upstream_ should be a vector instead of a single token ?
  edm::EDGetTokenT<MPIToken> const upstream_;  // MPIToken used to establish the communication channel
  edm::EDPutTokenT<MPIToken> const token_;  // copy of the MPIToken that may be used to implement an ordering relation
  std::vector<Entry> products_;             // data to be read over the channel and put into the Event
  int32_t const instance_;                  // instance used to identify the source-destination pair

  std::shared_ptr<ProductMetadataBuilder> received_meta_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MPIReceiver);