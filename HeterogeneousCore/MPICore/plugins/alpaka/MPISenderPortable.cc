// C++ include files
#include <string>
#include <vector>

// CMSSW include files
#include "DataFormats/Common/interface/PathStateToken.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/WrapperBaseHandle.h"
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
#include "HeterogeneousCore/TrivialSerialisation/interface/ReaderBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/ReaderBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Inherit from ProducerBase. This is so we have access to the EDMetadata,
  // which we need for synchronization
  class MPISenderPortable : public ProducerBase<edm::stream::EDProducer, edm::ExternalWork> {
  public:
    MPISenderPortable(edm::ParameterSet const& config)
        : ProducerBase<edm::stream::EDProducer, edm::ExternalWork>(config),
          upstream_(consumes<MPIToken>(config.getParameter<edm::InputTag>("upstream"))),
          token_(this->producesCollector().template produces<MPIToken>()),
          instance_(config.getParameter<int32_t>("instance")) {
      // instance 0 is reserved for the MPIController / MPISource pair instance
      // values greater than 255 may not fit in the MPI tag
      if (instance_ < 1 or instance_ > 255) {
        throw cms::Exception("InvalidValue")
            << "Invalid MPISenderPortable instance value, please use a value between 1 and 255";
      }

      // Products are configured as a VPSet with explicit type/label/instance, rather than
      // vstring patterns + callWhenNewProductsRegistered. This is because device types are looked up
      // via the human-readable name in SerialiserFactoryDevice, not via branch name pattern matching.
      auto const& products = config.getParameter<std::vector<edm::ParameterSet>>("products");
      products_.reserve(products.size());
      for (auto const& product : products) {
        auto const& type = product.getParameter<std::string>("type");
        auto const& label = product.getParameter<std::string>("label");
        auto const& instance = product.getParameter<std::string>("instance");

        Entry entry;
        entry.typeName = type;

        // This module does not (yet?) support sending PathStateToken, or any
        // other host product. The path status is propagated through productCount.
        if (type == "edm::PathStateToken") {
          entry.typeID = edm::TypeID(typeid(edm::PathStateToken));
          entry.token = this->consumes(edm::TypeToGet{entry.typeID, edm::PRODUCT_TYPE}, edm::InputTag{label, instance});
          products_.emplace_back(std::move(entry));
          continue;
        }

        // Get a device serialiser from the human-readable type that we got from
        // the python config. Currently, there is not fallback if a *device*
        // serialiser for a certain type is not found. Host products must be
        // transfered through the non-portable MPISender.
        std::unique_ptr<ngt::SerialiserBase> serialiser{ngt::SerialiserFactoryDevice::get()->tryToCreate(type)};
        if (!serialiser) {
          throw cms::Exception("MPISenderPortable") << "No device serialiser found for type '" << type
                                                    << "'. Only device products are supported by MPISenderPortable. "
                                                    << "Use MPISender for host products.";
        }

        // Get the edm::TypeID type from the serialiser, which we will later
        // need to construct the edm::Handle where we will put the product we get
        // from the event. This typeID is the product type wrapped in
        // edm::DeviceProduct, and includes the alpaka device.
        edm::TypeID typeID{serialiser->productTypeID()};
        entry.typeID = typeID;

        entry.token = this->consumes(edm::TypeToGet{typeID, edm::PRODUCT_TYPE}, edm::InputTag{label, instance});

        LogDebug("MPISenderPortable") << "send type \"" << typeID << "\" (" << type << "), label \"" << label
                                      << "\" instance \"" << instance << "\" over MPI channel instance " << instance_;

        products_.emplace_back(std::move(entry));
      }
    }

    void acquire(edm::Event const& event, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder holder) final {
      // This would normally be done in the SynchronizingEDProducer.h . Here we do it manually.
      detail::EDMetadataAcquireSentry sentry(event.streamID(), std::move(holder), this->synchronize());
      metadata_ = sentry.metadata();

      const MPIToken& token = event.get(upstream_);

      // Products of type edm::PathStateToken are not sent.
      size_t productCount = 0;
      for (auto const& entry : products_) {
        if (entry.typeName != "edm::PathStateToken")
          ++productCount;
      }
      auto meta = std::make_shared<ProductMetadataBuilder>(productCount);
      is_active_ = true;

      // Prepare a "vector of readers". Readers that will be created here and reused in produce().
      readers_.clear();
      readers_.resize(productCount);

      size_t index = 0;
      for (auto const& entry : products_) {
        edm::Handle<edm::WrapperBase> handle(entry.typeID.typeInfo());
        event.getByToken(entry.token, handle);

        // Product count -1 indicates that the event has been filtered out, i.e. that this path is not active.
        if (not handle.isValid() and entry.typeName == "edm::PathStateToken") {
          meta->setProductCount(-1);
          is_active_ = false;
          break;
        }

        // edm::PathStateToken is not sent.
        if (entry.typeName == "edm::PathStateToken") {
          continue;
        }

        if (handle.isValid()) {
          edm::WrapperBase const* wrapper = handle.product();

          // Get the device serialiser for the product type, and then the reader from the serialiser.
          std::unique_ptr<ngt::SerialiserBase> serialiser{
              ngt::SerialiserFactoryDevice::get()->tryToCreate(entry.typeName)};

          // TODO: I'm not very convinced that "hardcoding" the tryReuseQueue=true here is fine.
          auto reader = serialiser->reader(*wrapper, *metadata_, /* tryReuseQueue */ true);
          ::ngt::AnyBuffer buffer = reader->parameters();
          meta->addTrivialCopy(buffer.data(), buffer.size_bytes());

          // Store the reader, so we don't need to recreate it in produce().
          readers_[index] = std::move(reader);
        } else {
          // handle missing product
          meta->addMissing();
        }
        index++;
      }

      // Use a non-blocking MPI call (MPI_Isend) to send the metadata. A few
      // notes on this approach: 1) We use MPI_Isend, instead of MPI_Issend, to
      // send the metadata. 2) This means that the MPI_Wait(), that will be
      // called in produce(), can return *before* the send has actually started.
      // 3) This is fine, due to the *nonovertaking* property of MPI messages
      // (see the MPI standard). 4) We need to keep the metadata buffer (i.e.
      // the ProductMetadataBuilder) alive until the send completes, which is
      // why we move it to a member variable.
      metadata_builder_ = std::move(meta);
      mpi_request_ = token.channel()->sendMetadataAsync(instance_, metadata_builder_);

      metadata_ = sentry.finish();  // see SynchronizingEDProducer.h
    }

    void produce(edm::Event& event, edm::EventSetup const&) final {
      // see SynchronizingEDProducer.h again
      detail::EDMetadataSentry sentry(std::move(metadata_), this->synchronize());

      MPIToken token = event.get(upstream_);

      // Wait for the non-blocking metadata send started in acquire().
      MPIChannel::waitMetadata(mpi_request_);
      metadata_builder_.reset();

      if (!is_active_) {
        event.emplace(token_, token);

        // Emplace the backend into the event and finish (see SynchronizingEDProducer.h)
        this->putBackend(event);
        sentry.finish(false);
        return;
      }

      // Ask each reader to send its product.
      for (auto const& reader : readers_) {
        if (reader) {
          // if the product is not an edm::PathStateToken and the product's handle was found to be valid in acquire()
          token.channel()->sendTrivialCopyProduct(instance_, *reader);
        }
      }

      // write a shallow copy of the channel to the output, so other modules can consume it
      // to indicate that they should run after this
      event.emplace(token_, token);

      this->putBackend(event);

      // "false" because no async work has been launched in this produce()
      sentry.finish(false);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      descriptions.setComment(
          "This module can consume arbitrary device event products and copy them "
          "to an \"MPIReceiverPortable\" module in a separate CMSSW job. "
          "For host products, use \"MPISender\" instead.");

      edm::ParameterSetDescription product;
      product.add<std::string>("type")->setComment(
          "Type name of the device product to be sent, using the human-readable type alias "
          "(e.g. \"hcal::RecHitDeviceCollection\").");
      product.add<std::string>("label")->setComment("Module label of the producer.");
      product.add<std::string>("instance", "")->setComment("Product instance name.");

      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("upstream", {"source"})
          ->setComment(
              "MPI communication channel. Can be an \"MPIController\", \"MPISource\", \"MPISender\" or "
              "\"MPIReceiver\". Passing an \"MPIController\" or \"MPISource\" only identifies the pair of local and "
              "remote application that communicate. Passing an \"MPISender\" or \"MPIReceiver\" in addition imposes a "
              "scheduling dependency.");
      desc.addVPSet("products", product, {})
          ->setComment("Device products to be consumed and copied over to a separate CMSSW job. ");
      desc.add<int32_t>("instance", 0)
          ->setComment("A value between 1 and 255 used to identify a matching pair of \"MPISender\"/\"MPIReceiver\".");

      descriptions.addWithDefaultLabel(desc);
    }

  private:
    struct Entry {
      std::string typeName;  // human-readable type name from config (for device serialiser lookup)
      edm::TypeID typeID;    // product type as registered in the framework
      edm::EDGetToken token;
    };

    edm::EDGetTokenT<MPIToken> const upstream_;  // MPIToken used to establish the communication channel
    edm::EDPutTokenT<MPIToken> const token_;  // copy of the MPIToken that may be used to implement an ordering relation
    std::vector<Entry> products_;             // types and tokens corresponding to the products
    int32_t const instance_;                  // instance used to identify the source-destination pair

    // vector of readers, first created in acquire() and then reused in produce().
    std::vector<std::unique_ptr<const ::ngt::ReaderBase>> readers_;
    bool is_active_ = true;

    // MPI_Request that we get back when we call the non-blocking send for the metadata in acquire(). We wait on it in produce().
    MPI_Request mpi_request_ = MPI_REQUEST_NULL;
    std::shared_ptr<ProductMetadataBuilder> metadata_builder_;

    // the EDMetadata object that we pass to the readers to manage the device queue and synchronisation.
    std::shared_ptr<EDMetadata> metadata_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(MPISenderPortable);
