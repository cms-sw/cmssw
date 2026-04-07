/*
 * This Alpaka EDProducer will clone all the device event products declared in
 * its configuration, using the plugin-based NGT trivial serialisation.
 *
 * This module works only with device products that have a device serialiser
 * registered in SerialiserFactoryDevice.
 *
 * Products are configured as a VPSet with type, label, and instance.
 * The type must be the human-readable type alias used to register the device
 * serialiser (e.g. "portabletest::TestDeviceCollection").
 */

// C++ include files
#include <cassert>
#include <memory>
#include <string>
#include <vector>

// CMSSW include files
#include "DataFormats/AlpakaCommon/interface/alpaka/EDMetadata.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/WrapperBaseHandle.h"
#include "FWCore/Framework/interface/WrapperBaseOrphanHandle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ProducerBase.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/ReaderBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/WriterBase.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt {

  class GenericClonerDevice : public ProducerBase<edm::stream::EDProducer> {
  public:
    explicit GenericClonerDevice(edm::ParameterSet const& config);
    ~GenericClonerDevice() override = default;

    void produce(edm::Event& event, edm::EventSetup const&) final;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    struct Entry {
      std::string typeName;  // human-readable type name, coming from the config, for serialiser lookup.
      edm::TypeID typeID;    // product type as registered in the framework
      edm::EDGetToken getToken;
      edm::EDPutToken putToken;
    };

    std::vector<Entry> eventProducts_;
    bool verbose_;
  };

  GenericClonerDevice::GenericClonerDevice(edm::ParameterSet const& config)
      : ProducerBase<edm::stream::EDProducer>(config), verbose_(config.getUntrackedParameter<bool>("verbose")) {
    auto const& products = config.getParameter<std::vector<edm::ParameterSet>>("eventProducts");
    eventProducts_.reserve(products.size());

    for (auto const& product : products) {
      auto const& type = product.getParameter<std::string>("type");
      auto const& label = product.getParameter<std::string>("label");
      auto const& instance = product.getParameter<std::string>("instance");

      // Get a device serialiser from the human-readable type that we got from
      // the python config.
      std::unique_ptr<ngt::SerialiserBase> serialiser{ngt::SerialiserFactoryDevice::get()->tryToCreate(type)};
      if (!serialiser) {
        throw edm::Exception(edm::errors::Configuration)
            << "No device serialiser found for type '" << type
            << "'. Only device products with a registered device serialiser are supported by GenericClonerDevice.";
      }

      Entry entry;
      entry.typeName = type;

      // Get the edm::TypeID from the serialiser. For asynchronous backends, this is the product type
      // wrapped in edm::DeviceProduct; for serial_sync, this is the product type directly.
      entry.typeID = edm::TypeID{serialiser->productTypeID()};

      entry.getToken = this->consumes(edm::TypeToGet{entry.typeID, edm::PRODUCT_TYPE}, edm::InputTag{label, instance});
      entry.putToken = this->producesCollector().produces(entry.typeID, instance);

      if (verbose_) {
        edm::LogInfo("GenericClonerDevice") << "will clone device product of type '" << type << "', label '" << label
                                            << "', instance '" << instance << "'";
      }

      eventProducts_.emplace_back(std::move(entry));
    }
  }

  void GenericClonerDevice::produce(edm::Event& event, edm::EventSetup const& /*unused*/) {
    // The EDMetadata needs to be handled manually, as would normally be done in
    // SynchronizingEDProducer.h
    detail::EDMetadataSentry sentry(event.streamID(), this->synchronize());

    for (size_t i = 0; i < eventProducts_.size(); ++i) {
      auto const& entry = eventProducts_[i];

      // Get the product from the Event, as a WrapperBase pointer.
      edm::Handle<edm::WrapperBase> handle(entry.typeID.typeInfo());
      event.getByToken(entry.getToken, handle);
      edm::WrapperBase const* wrapper = handle.product();

      // Get a device serialiser and initialise the corresponding Reader and Writer.
      std::unique_ptr<ngt::SerialiserBase> serialiser{ngt::SerialiserFactoryDevice::get()->tryToCreate(entry.typeName)};
      auto reader = serialiser->reader(*wrapper, *sentry.metadata());
      auto writer = serialiser->writer();

      // Initialise the clone with the queue and the parameters from the source.
      writer->initialize(sentry.metadata()->queue(), reader->parameters());

      // Copy the source regions to the target.
      auto targets = writer->regions();
      auto sources = reader->regions();

      assert(sources.size() == targets.size());
      for (size_t j = 0; j < sources.size(); ++j) {
        assert(sources[j].data() != nullptr);
        assert(targets[j].data() != nullptr);
        assert(targets[j].size_bytes() == sources[j].size_bytes());
        alpaka::memcpy(sentry.metadata()->queue(),
                       cms::alpakatools::make_device_view(sentry.metadata()->queue(), targets[j]),
                       cms::alpakatools::make_device_view(sentry.metadata()->queue(), sources[j]));
      }

      // Finalize the clone after the copy, if the type requires it.
      writer->finalize();

      // Move the clone into the Event.
      event.put(entry.putToken, writer->get(sentry.metadata()));
    }

    this->putBackend(event);
    sentry.finish(true);
  }

  void GenericClonerDevice::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    descriptions.setComment(
        "This Alpaka EDProducer will clone all the device event products declared by its configuration, "
        "using the device TrivialSerialisation mechanism.");

    edm::ParameterSetDescription product;
    product.add<std::string>("type")->setComment(
        "Type name of the device product to be cloned, using the human-readable type alias "
        "(e.g. \"portabletest::TestDeviceCollection\").");
    product.add<std::string>("label")->setComment("Module label of the producer.");
    product.add<std::string>("instance", "")->setComment("Product instance name.");

    edm::ParameterSetDescription desc;
    desc.addVPSet("eventProducts", product, {})->setComment("Device products to be cloned.");
    desc.addUntracked<bool>("verbose", false)->setComment("Print the type names of the products that will be cloned.");

    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

DEFINE_FWK_ALPAKA_MODULE(ngt::GenericClonerDevice);
