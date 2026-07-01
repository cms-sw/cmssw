// C++ include files
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <TBufferFile.h>
#include <TClass.h>
#include <alpaka/alpaka.hpp>
#include <mpi.h>

// CMSSW include files
#include "DataFormats/AlpakaCommon/interface/alpaka/EDMetadata.h"
#include "DataFormats/Common/interface/PathStateToken.h"
#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/WrapperBaseHandle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ProducerBase.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/chooseDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/MPICore/interface/MPIChannel.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/ReaderBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserFactory.h"
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

      auto const& products = config.getParameter<std::vector<edm::ParameterSet>>("products");
      products_.reserve(products.size());
      for (auto const& product : products) {
        auto const& type = product.getParameter<std::string>("type");
        auto const& src = product.getParameter<edm::InputTag>("src");

        Entry entry;
        entry.typeName = type;

        // PathStateToken is not transferred over MPI; the path status is
        // propagated through productCount, which will be set to -1 if the path
        // is inactive.
        if (type == "edm::PathStateToken") {
          entry.typeID = edm::TypeID(typeid(edm::PathStateToken));
          entry.token = this->consumes(edm::TypeToGet{entry.typeID, edm::PRODUCT_TYPE}, src);
          products_.emplace_back(std::move(entry));
          continue;
        }

        // Lookup the right serialiser. In order of preference:
        // SerialiserFactoryDevice, SerialiserFactory, ROOT Serialisation.
        //
        // Check 1: Construct the mangled typeid of a device type from the type
        // alias given in the config (e.g. "sistrip::SiStripClusterDevice"), and
        // use this mangled type to look up a device serialiser.
        LogDebug("MPISenderPortable") << "looking for device serialiser for type \"" << type << "\"";
        edm::TypeWithDict const deviceTypeTwd =
            edm::TypeWithDict::byName(std::string(EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)) + "::" + type);
        std::unique_ptr<ngt::SerialiserBase> deviceSerialiser;
        if (deviceTypeTwd.typeInfo() != typeid(void)) {
          deviceSerialiser = ngt::SerialiserFactoryDevice::get()->tryToCreate(deviceTypeTwd.typeInfo().name());
        }

        if (deviceSerialiser) {
          LogDebug("MPISenderPortable") << "found device serialiser for type \"" << type << "\"";
          // Get the edm::TypeID type from the serialiser, which we will later
          // need to construct the edm::Handle where we will put the product we get
          // from the event. This typeID is the product type wrapped in
          // edm::DeviceProduct, and includes the alpaka device.
          edm::TypeID typeID{deviceSerialiser->productTypeID()};
          hasDeviceProducts_ = true;
          entry.typeID = typeID;
          entry.token = this->consumes(edm::TypeToGet{typeID, edm::PRODUCT_TYPE}, src);
          entry.deviceSerialiser = std::move(deviceSerialiser);

          LogDebug("MPISenderPortable") << "send device type \"" << typeID << "\" (" << type << "), label \""
                                        << src.label() << "\" instance \"" << src.instance()
                                        << "\" over MPI channel instance " << instance_;

          products_.emplace_back(std::move(entry));
          continue;
        }

        // Check 2: Lookup a host serialiser registered in the host
        // SerialiserFactory.
        edm::TypeWithDict twd = edm::TypeWithDict::byName(type);
        std::unique_ptr<::ngt::SerialiserBase> hostSerialiser;

        LogDebug("MPISenderPortable") << "looking for host serialiser for type \"" << type << "\"";
        if (twd.typeInfo() != typeid(void)) {
          hostSerialiser = ::ngt::SerialiserFactory::get()->tryToCreate(twd.typeInfo().name());
        }
        if (hostSerialiser) {
          LogDebug("MPISenderPortable") << "found host serialiser for type \"" << type << "\"";
          entry.typeID = edm::TypeID{twd.typeInfo()};
          entry.token = this->consumes(edm::TypeToGet{entry.typeID, edm::PRODUCT_TYPE}, src);
          entry.hostSerialiser = std::move(hostSerialiser);

          LogDebug("MPISenderPortable") << "send host type \"" << entry.typeID << "\" (" << type << "), label \""
                                        << src.label() << "\" instance \"" << src.instance()
                                        << "\" over MPI channel instance " << instance_;

          products_.emplace_back(std::move(entry));
          continue;
        }

        // Check 3: Fall back to ROOT serialisation, if a ROOT dictionary is
        // found for this type
        edm::TypeWithDict wrappedTwd = edm::TypeWithDict::byName("edm::Wrapper<" + type + ">");
        LogDebug("MPISenderPortable") << "looking for ROOT serialisation of type \"" << type << "\"";
        if (twd.typeInfo() == typeid(void) || !wrappedTwd.getClass()) {
          throw cms::Exception("MPISenderPortable")
              << "No serialisation mechanism (device or host TrivialSerialisation, or ROOT dictionaries) found for "
                 "type '"
              << type
              << "'. Either register a serialiser via DEFINE_TRIVIAL_SERIALISER_PLUGIN or "
                 "DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN, or make sure a ROOT dictionary exists for this type.";
        }
        LogDebug("MPISenderPortable") << "found ROOT dictionary for type \"" << type << "\"";

        entry.typeID = edm::TypeID{twd.typeInfo()};
        entry.token = this->consumes(edm::TypeToGet{entry.typeID, edm::PRODUCT_TYPE}, src);
        entry.wrappedType = wrappedTwd;

        LogDebug("MPISenderPortable") << "send ROOT type \"" << entry.typeID << "\" (" << type << "), label \""
                                      << src.label() << "\" instance \"" << src.instance()
                                      << "\" over MPI channel instance " << instance_;

        products_.emplace_back(std::move(entry));
      }

      LogDebug("MPISenderPortable") << "configured to send " << products_.size()
                                    << " products over MPI channel instance " << instance_;
    }

    void acquire(edm::Event const& event, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder holder) final {
      MPIToken const& token = event.get(upstream_);

      size_t productCount = 0;
      for (auto const& entry : products_)
        if (entry.typeName != "edm::PathStateToken")
          ++productCount;

      auto productMetadata = std::make_shared<ProductMetadataBuilder>(productCount);
      bool isActive = true;

      struct DataToBeSent {
        using Regions = std::vector<std::span<const std::byte>>;
        std::vector<Regions> pendingRegions;
        // Anything we pass to runAsync needs to be copyable, so we wrap
        // rootBuffer in this struct and pass to runAsync the
        // std::make_shared<DataToBeSent>() below.
        std::unique_ptr<TBufferFile> rootBuffer;
      };
      auto toBeSent = std::make_shared<DataToBeSent>();
      toBeSent->pendingRegions.reserve(productCount);

      // The EDMetadata the device serialisers need to access a device product T
      // from its wrapped form.
      std::shared_ptr<EDMetadata> deviceMetadata;
      if (hasDeviceProducts_) {
        deviceMetadata = std::make_shared<EDMetadata>(detail::chooseDevice(event.streamID()));
      }

      for (auto const& entry : products_) {
        edm::Handle<edm::WrapperBase> handle(entry.typeID.typeInfo());
        event.getByToken(entry.token, handle);

        if (not handle.isValid() and entry.typeName == "edm::PathStateToken") {
          productMetadata->setProductCount(-1);
          isActive = false;
          break;
        }
        if (entry.typeName == "edm::PathStateToken")
          continue;

        if (handle.isValid()) {
          edm::WrapperBase const* wrapper = handle.product();
          // extract memory regions
          if (entry.deviceSerialiser) {
            // If the product is on device
            auto reader = entry.deviceSerialiser->reader(*wrapper, *deviceMetadata);
            ::ngt::AnyBuffer buffer = reader->parameters();
            productMetadata->addTrivialCopy(buffer.data(), buffer.size_bytes());
            toBeSent->pendingRegions.push_back(reader->regions());
          } else if (entry.hostSerialiser) {
            // If the product is on host and we have a serialiser for it
            auto reader = entry.hostSerialiser->reader(*wrapper);
            ::ngt::AnyBuffer buffer = reader->parameters();
            productMetadata->addTrivialCopy(buffer.data(), buffer.size_bytes());
            toBeSent->pendingRegions.push_back(reader->regions());
          } else {
            // If the product is serialised via ROOT
            TClass* cls = entry.wrappedType.getClass();
            if (!cls)
              throw cms::Exception("MPISenderPortable") << "Failed to get TClass for type: " << entry.typeName;
            if (!toBeSent->rootBuffer)
              toBeSent->rootBuffer = std::make_unique<TBufferFile>(TBuffer::kWrite);
            size_t prevLen = toBeSent->rootBuffer->Length();
            cls->Streamer(const_cast<void*>(static_cast<void const*>(wrapper)), *toBeSent->rootBuffer);
            productMetadata->addSerialized(toBeSent->rootBuffer->Length() - prevLen);
          }
        } else {
          productMetadata->addMissing();
        }
      }

      // Send metadata immediately; wait for its completion in runAsync before
      // sending product data. productMetadata will be passed to runAsync so it
      // is kept alive until sendMetadataAsync completes.
      auto productMetadataRequest =
          std::make_shared<MPI_Request>(token.channel()->sendMetadataAsync(instance_, productMetadata));

      // Lambda that waits for device work and the metadata send, then sends
      // all data products, to be passed to runAsync.
      auto sendData =
          [token, instance = instance_, productMetadata, productMetadataRequest, toBeSent, isActive, deviceMetadata]() {
            if (deviceMetadata) {
              alpaka::wait(deviceMetadata->queue());
            }
            MPIChannel::waitMetadata(*productMetadataRequest);
            if (isActive) {
              if (toBeSent->rootBuffer)
                token.channel()->sendBuffer(toBeSent->rootBuffer->Buffer(),
                                            toBeSent->rootBuffer->Length(),
                                            instance,
                                            EDM_MPI_SendSerializedProduct);
              for (auto const& regions : toBeSent->pendingRegions)
                token.channel()->sendTrivialCopyProduct(instance, regions);
            }
          };

      edm::Service<edm::Async> asyncService;
      asyncService->runAsync(
          std::move(holder), std::move(sendData), []() { return "Calling MPISenderPortable::acquire()"; });
    }

    void produce(edm::Event& event, edm::EventSetup const&) final {
      // write a shallow copy of the channel to the output, so other modules can
      // consume it to indicate that they should run after this
      MPIToken token = event.get(upstream_);
      event.emplace(token_, token);

      this->putBackend(event);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      descriptions.setComment(
          "This module can consume arbitrary host or device event products and copy them "
          "to an \"MPIReceiverPortable\" module in a separate CMSSW job. "
          "Products are serialised using the device serialiser (if available), the host "
          "serialiser (if available), or ROOT as a fallback.");

      edm::ParameterSetDescription product;
      product.add<std::string>("type")->setComment(
          "Type alias of the device product without the ALPAKA_ACCELERATOR_NAMESPACE prefix "
          "(e.g. \"sistrip::SiStripClusterDevice\"). For host and ROOT products, the plain C++ type name.");
      product.add<edm::InputTag>("src")->setComment(
          "InputTag identifying the product to consume: label is the producer module label, "
          "instance is the product instance name.");

      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("upstream", {"source"})
          ->setComment(
              "MPI communication channel. Can be an \"MPIController\", \"MPISource\", or "
              "\"MPISenderPortable\"/\"MPIReceiverPortable\". Passing an \"MPIController\" or \"MPISource\" "
              "only identifies the pair of local and remote applications. Passing a sender or receiver "
              "in addition imposes a scheduling dependency.");
      desc.addVPSet("products", product, {})
          ->setComment("Host or device products to be consumed and copied over to a separate CMSSW job.");
      desc.add<int32_t>("instance", 0)
          ->setComment(
              "A value between 1 and 255 used to identify a matching pair of "
              "\"MPISenderPortable\"/\"MPIReceiverPortable\".");

      descriptions.addWithDefaultLabel(desc);
    }

  private:
    struct Entry {
      std::string typeName;  // type name as written in the config
      edm::TypeID typeID;
      edm::EDGetToken token;

      std::unique_ptr<::ngt::SerialiserBase> hostSerialiser;
      std::unique_ptr<ngt::SerialiserBase> deviceSerialiser;
      edm::TypeWithDict wrappedType;
    };

    edm::EDGetTokenT<MPIToken> const upstream_;
    edm::EDPutTokenT<MPIToken> const token_;
    std::vector<Entry> products_;
    int32_t const instance_;
    bool hasDeviceProducts_ = false;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(MPISenderPortable);
