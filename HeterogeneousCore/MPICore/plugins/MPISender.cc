#include <string>
#include <string_view>
#include <vector>

// ROOT headers
#include <TBufferFile.h>
#include <TClass.h>

// CMSSW include files
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/ProductNamePattern.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/Framework/interface/GenericHandle.h"
#include "FWCore/Framework/interface/WrapperBaseHandle.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
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

#include <iostream>

// local include files
#include "api.h"

class MPISender : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  MPISender(edm::ParameterSet const& config)
      : upstream_(consumes<MPIToken>(config.getParameter<edm::InputTag>("upstream"))),
        token_(produces<MPIToken>()),
        patterns_(edm::productPatterns(config.getParameter<std::vector<std::string>>("products"))),
        instance_(config.getParameter<int32_t>("instance")),
        buffer_ = std::make_unique<TBufferFile>(TBuffer::kWrite),
        buffer_offset_(0),
        metadata_size_(0)
  {
    // instance 0 is reserved for the MPIController / MPISource pair
    // instance values greater than 255 may not fit in the MPI tag
    if (instance_ < 1 or instance_ > 255) {
      throw cms::Exception("InvalidValue") << "Invalid MPISender instance value, please use a value between 1 and 255";
    }

    products_.reserve(patterns_.size());

    callWhenNewProductsRegistered([this](edm::ProductDescription const& product) {
      static const std::string_view kPathStatus("edm::PathStatus");
      static const std::string_view kEndPathStatus("edm::EndPathStatus");

      switch (product.branchType()) {
        case edm::InEvent:
          if (product.className() == kPathStatus or product.className() == kEndPathStatus)
            return;
          for (auto const& pattern : patterns_) {
            if (pattern.match(product)) {
              Entry entry;
              entry.type = product.unwrappedType();
              entry.wrappedType = product.wrappedType();
              // TODO move this to EDConsumerBase::consumes() ?
              entry.token = this->consumes(
                  edm::TypeToGet{product.unwrappedTypeID(), edm::PRODUCT_TYPE},
                  edm::InputTag{product.moduleLabel(), product.productInstanceName(), product.processName()});

              edm::LogVerbatim("MPISender")
                  << "send product \"" << product.friendlyClassName() << '_' << product.moduleLabel() << '_'
                  << product.productInstanceName() << '_' << product.processName() << "\" of type \""
                  << entry.type.name() << "\" over MPI channel instance " << instance_;

              edm::Handle<edm::WrapperBase> handle(entry.type.typeInfo());

              edm::WrapperBase const* wrapper = handle.product();

              // parameter getting does not work in constructor, i suppose wee need an event for this
              
              // if (wrapper->hasTrivialCopyTraits()) {
              //   metadata_size_ += 9;
              // } else {
                // parameter getting does not work in constructor, i suppose wee need an event for this

                // edm::AnyBuffer buffer = wrapper->trivialCopyParameters();
                // metadata_size_ += buffer.size_bytes();
              //   metadata_size_ += 9;
              //   metadata_size_ += 24;
              // }
              products_.emplace_back(std::move(entry));
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
    MPIToken token = event.get(upstream_);
    auto meta = std::make_shared<ProductMetadataBuilder>(products_.size()*24);
    size_t index = 0;
    // this seems to work fine, but does this vector indeed persist between acquire() and produce()?
    // serializedBuffers_.clear();
    buffer_->Reset();
    buffer_offset_ = 0;

    // estimate buffer size in the constructor

    for (auto const& entry : products_) {
      // Get the product
      edm::Handle<edm::WrapperBase> handle(entry.type.typeInfo());
      event.getByToken(entry.token, handle);

      if (handle.isValid()) {
        edm::WrapperBase const* wrapper = handle.product();
        
        // can there be such product without these properties?
        // how should the case be handled if yes? 
        if (wrapper->hasTrivialCopyTraits()) {
          edm::AnyBuffer buffer = wrapper->trivialCopyParameters();
          meta->addTrivialCopy(buffer.data(), buffer.size_bytes());
        } else {
          TClass* cls = entry.wrappedType.getClass();
          if (!cls) {
            throw cms::Exception("MPISender") << "Failed to get TClass for type: " << entry.type.name();
          }

          size_t bufLen = serializeAndStoreBuffer_(index, cls, wrapper);
          meta->addSerialized(bufLen);
        }

      } else {
        // handle missing product
        meta->addMissing();
      }
      index++;
    }

    // Submit sending of all products to run in the additional asynchronous threadpool
    edm::Service<edm::Async> as;
    as->runAsync(
        std::move(holder),
        [this, token, meta = std::move(meta)]() {
           token.channel()->sendMetadata(instance_, meta); },
        []() { return "Calling MPISender::acquire()"; });
  }

  void produce(edm::Event& event, edm::EventSetup const&) final {
    MPIToken token = event.get(upstream_);
    size_t index = 0;
    size_t serializedIndex = 0;

    // std::cerr << "MPISender::produce(): "
    //           << serializedBuffers_.size() << " serialized product(s) stored.\n";
    // for (auto const& [index, buffer] : serializedBuffers_) {
    //   std::cerr << "  - index: " << index << ", buffer size: " << buffer->Length() << " bytes\n";
    // }

    for (auto const& entry : products_) {
      edm::Handle<edm::WrapperBase> handle(entry.type.typeInfo());
      event.getByToken(entry.token, handle);
      edm::WrapperBase const* wrapper = handle.product();
      // we don't send missing products
      if (handle.isValid()) {
        if (wrapper->hasTrivialCopyTraits()) {
          token.channel()->sendTrivialCopyProduct_(instance_, wrapper);
        } else {
          assert(serializedBuffers_[serializedIndex].first == index && "mismatch between expected and factual serialised index in produce()");
          token.channel()->sendBuffer(serializedBuffers_[serializedIndex].second->Buffer(), serializedBuffers_[serializedIndex].second->Length(), instance_, EDM_MPI_SendSerializedProduct);
          serializedIndex++;
        }
        index++;
      }
    }
    // write a shallow copy of the channel to the output, so other modules can consume it
    // to indicate that they should run after this
    event.emplace(token_, token);
  }

private:
  
  size_t serializeAndStoreBuffer_(size_t index, TClass* type, void const* product) {
    // auto buffer = std::make_shared<TBufferFile>(TBuffer::kWrite);
    // buf->ResetMap();
    type->Streamer(const_cast<void*>(product), *buffer_);
    // serializedBuffers_.emplace_back(index, buffer);
    size_t prod_size = buf->Length() - buffer_offset_;
    buffer_offset_ = buf->Length(); 
    return prod_size;
  }

  struct Entry {
    edm::TypeWithDict type;
    edm::TypeWithDict wrappedType;
    edm::EDGetToken token;
  };

  // TODO consider if upstream_ should be a vector instead of a single token ?
  edm::EDGetTokenT<MPIToken> const upstream_;  // MPIToken used to establish the communication channel
  edm::EDPutTokenT<MPIToken> const token_;  // copy of the MPIToken that may be used to implement an ordering relation
  std::vector<edm::ProductNamePattern> patterns_;  // branches to read from the Event and send over the MPI channel
  std::vector<Entry> products_;                    // types and tokens corresponding to the branches
  int32_t const instance_;                         // instance used to identify the source-destination pair
  // std::vector<std::pair<size_t, std::shared_ptr<TBufferFile>>> serializedBuffers_;  // buffers per serialized product
  std::unique_ptr<TBufferFile> buffer_;
  size_t buffer_offset_;
  size_t metadata_size_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MPISender);