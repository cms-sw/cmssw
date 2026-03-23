// C++ include files
#include <cassert>
#include <string>
#include <vector>

// CMSSW include files
#include "DataFormats/Common/interface/PathStateToken.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/WrapperBaseOrphanHandle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadata.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataAcquireSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ProducerBase.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"
#include "HeterogeneousCore/MPICore/interface/api.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/WriterBase.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Inherit from ProducerBase. This is so we have access to the EDMetadata,
  // which we need for synchronization
  class MPIReceiverPortable : public ProducerBase<edm::stream::EDProducer, edm::ExternalWork> {
    using Base = ProducerBase<edm::stream::EDProducer, edm::ExternalWork>;

  public:
    MPIReceiverPortable(edm::ParameterSet const& config)
        : Base(config),
          upstream_(consumes<MPIToken>(config.getParameter<edm::InputTag>("upstream"))),
          token_(this->producesCollector().template produces<MPIToken>()),
          instance_(config.getParameter<int32_t>("instance")) {
      // instance 0 is reserved for the MPIController / MPISource pair instance
      // values greater than 255 may not fit in the MPI tag
      if (instance_ < 1 or instance_ > 255) {
        throw cms::Exception("InvalidValue")
            << "Invalid MPIReceiverPortable instance value, please use a value between 1 and 255";
      }

      // Products are configured as a VPSet with explicit type/label/instance, rather than
      // vstring patterns + callWhenNewProductsRegistered. This is because device types are looked up
      // via the human-readable name in SerialiserFactoryDevice, not via branch name pattern matching.
      auto const& products = config.getParameter<std::vector<edm::ParameterSet>>("products");
      products_.reserve(products.size());
      for (auto const& product : products) {
        auto const& type = product.getParameter<std::string>("type");
        auto const& label = product.getParameter<std::string>("label");

        Entry entry;
        entry.typeName = type;

        // This module does not (yet?) support receiving PathStateToken, or any
        // other host product. The path status is propagated through productCount.
        if (type == "edm::PathStateToken") {
          entry.token = this->producesCollector().template produces<edm::PathStateToken>();
          products_.emplace_back(std::move(entry));
          continue;
        }

        // Get a device serialiser from the human-readable type that we got from
        // the python config. Currently, there is not fallback if a *device*
        // serialiser for a certain type is not found. Host products must be
        // transfered through the non-portable MPISender.
        std::unique_ptr<ngt::SerialiserBase> serialiser{ngt::SerialiserFactoryDevice::get()->tryToCreate(type)};
        if (!serialiser) {
          throw cms::Exception("MPIReceiverPortable")
              << "No device serialiser found for type '" << type
              << "'. Only device products are supported by MPIReceiverPortable. "
              << "Use MPIReceiver for host products.";
        }

        // Get the edm::TypeID type from the serialiser, which we will later
        // need to construct the edm::Handle where we will put the product we get
        // from the event. This typeID is the product type wrapped in
        // edm::DeviceProduct, and includes the alpaka device.
        edm::TypeID typeID{serialiser->productTypeID()};

        entry.token = this->producesCollector().produces(typeID, label);

        LogDebug("MPIReceiverPortable") << "receive type \"" << typeID << "\" (" << type << ") for label \"" << label
                                        << "\" over MPI channel instance " << instance_;

        products_.emplace_back(std::move(entry));
      }
    }

    void acquire(edm::Event const& event, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder holder) final {
      // This would normally be done in the SynchronizingEDProducer.h . Here we do it manually.
      detail::EDMetadataAcquireSentry sentry(event.streamID(), std::move(holder), this->synchronize());
      metadata_ = sentry.metadata();

      const MPIToken& token = event.get(upstream_);

      received_meta_ = std::make_shared<ProductMetadataBuilder>();

      // Start a non-blocking MPI receive for the metadata. The corresponding MPI_Wait is called in produce().
      token.channel()->receiveMetadataAsync(instance_, received_meta_);

      metadata_ = sentry.finish();
    }

    void produce(edm::Event& event, edm::EventSetup const&) final {
      // see SynchronizingEDProducer.h
      detail::EDMetadataSentry sentry(std::move(metadata_), this->synchronize());

      // read the MPIToken used to establish the communication channel
      MPIToken token = event.get(upstream_);

      // Wait for the non-blocking metadata receive started in acquire().
      received_meta_->waitReceiveMetadata();

#ifdef EDM_ML_DEBUG
      // dump the summary of metadata
      received_meta_->debugPrintMetadataSummary();
#endif

      // if filter was false before the sender, receive nothing
      if (received_meta_->productCount() == -1) {
        event.emplace(token_, token);

        // Emplace the backend into the event and finish (see SynchronizingEDProducer.h)
        this->putBackend(event);
        sentry.finish(false);
        return;
      }

      bool asyncWorkLaunched = false;
      for (auto const& entry : products_) {
        // PathStateToken has no device serialiser and is not transferred by the portable MPI Sender and Receiver. Instead,
        // if the event was not filtered out before the sender (i.e. if productCount() was not -1 above), it is constructed here.
        if (entry.typeName == "edm::PathStateToken") {
          event.put(entry.token, std::make_unique<edm::PathStateToken>());
          continue;
        }

        auto product_meta = received_meta_->getNext();

        if (product_meta.kind == ProductMetadata::Kind::Missing) {
          edm::LogWarning("MPIReceiverPortable") << "Product " << entry.typeName << " was not received.";
          continue;
        }

        if (product_meta.kind == ProductMetadata::Kind::TrivialCopy) {
          std::unique_ptr<ngt::SerialiserBase> serialiser{
              ngt::SerialiserFactoryDevice::get()->tryToCreate(entry.typeName)};
          if (!serialiser) {
            throw cms::Exception("MPIReceiverPortable")
                << "No device serialiser found for type '" << entry.typeName << "'";
          }
          auto writer = serialiser->writer();
          ::ngt::AnyBuffer buffer = writer->uninitialized_parameters();
          assert(buffer.size_bytes() == product_meta.sizeMeta);
          std::memcpy(buffer.data(), product_meta.trivialCopyOffset, product_meta.sizeMeta);

          writer->initialize(sentry.metadata()->queue(), buffer);
          asyncWorkLaunched = true;
          token.channel()->receiveInitializedTrivialCopy(instance_, *writer);
          writer->finalize();

          // put the data into the Event
          event.put(entry.token, writer->get(sentry.metadata()));
        } else {
          throw cms::Exception("MPIReceiverPortable")
              << "Unexpected product metadata kind for device product '" << entry.typeName << "'. "
              << "Only TrivialCopy is supported for device products.";
        }
      }

      // write a shallow copy of the channel to the output, so other modules can consume it
      // to indicate that they should run after this
      event.emplace(token_, token);

      this->putBackend(event);
      sentry.finish(asyncWorkLaunched);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      descriptions.setComment(
          "This module can receive arbitrary device event products from an "
          "\"MPISenderPortable\" module in a separate CMSSW job, and produce them into the event. "
          "For host products, use \"MPIReceiver\" instead.");

      edm::ParameterSetDescription product;
      product.add<std::string>("type")->setComment(
          "Type name of the device product to be received, using the human-readable type alias "
          "(e.g. \"hcal::RecHitDeviceCollection\").");
      product.add<std::string>("label", "")
          ->setComment("Product instance label. Leave empty for the default instance.");

      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("upstream", {"source"})
          ->setComment(
              "MPI communication channel. Can be an \"MPIController\", \"MPISource\", \"MPISender\" or "
              "\"MPIReceiver\". Passing an \"MPIController\" or \"MPISource\" only identifies the pair of local and "
              "remote application that communicate. Passing an \"MPISender\" or \"MPIReceiver\" in addition imposes a "
              "scheduling dependency.");
      desc.addVPSet("products", product, {})->setComment("Device products to be received from a separate CMSSW job.");
      desc.add<int32_t>("instance", 0)
          ->setComment("A value between 1 and 255 used to identify a matching pair of \"MPISender\"/\"MPIReceiver\".");

      descriptions.addWithDefaultLabel(desc);
    }

  private:
    struct Entry {
      std::string typeName;  // human-readable type name from config (for device serialiser lookup)
      edm::EDPutToken token;
    };

    edm::EDGetTokenT<MPIToken> const upstream_;  // MPIToken used to establish the communication channel
    edm::EDPutTokenT<MPIToken> const token_;  // copy of the MPIToken that may be used to implement an ordering relation
    std::vector<Entry> products_;             // data to be read over the channel and put into the Event
    int32_t const instance_;                  // instance used to identify the source-destination pair

    std::shared_ptr<ProductMetadataBuilder> received_meta_;
    std::shared_ptr<EDMetadata> metadata_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortable);
