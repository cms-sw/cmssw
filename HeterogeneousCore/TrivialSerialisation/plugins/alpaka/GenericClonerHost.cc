/*
 * This Alpaka EDProducer will clone all the host event products declared in
 * its configuration, using the plugin-based NGT trivial serialisation.
 *
 * This module supports only host products that have a host serialiser
 * registered in SerialiserFactory, and a matching device serialiser registered
 * in SerialiserFactoryDevice under the same host type alias.
 *
 * Products are configured as a VPSet with type, label, and instance.
 * The type must be the human-readable host alias used by the host and device
 * serialiser plugin factories (e.g. "portabletest::TestHostCollection").
 */

// C++ include files
#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// CMSSW include files
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
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ProducerBase.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/ReaderBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserFactory.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/WriterBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt {

  class GenericClonerHost : public ProducerBase<edm::stream::EDProducer> {
  public:
    explicit GenericClonerHost(edm::ParameterSet const& config);
    ~GenericClonerHost() override = default;

    void produce(edm::Event& event, edm::EventSetup const&) final;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    struct Entry {
      std::string typeName;  // human-readable type name, coming from the config, for serialiser lookup.
      edm::TypeID typeID;    // Host product type registered in the framework.
      edm::EDGetToken getToken;
      edm::EDPutToken putToken;
      std::unique_ptr<::ngt::SerialiserBase> serialiser;
    };

    std::vector<Entry> eventProducts_;
    bool verbose_;
  };

  GenericClonerHost::GenericClonerHost(edm::ParameterSet const& config)
      : ProducerBase<edm::stream::EDProducer>(config), verbose_(config.getUntrackedParameter<bool>("verbose")) {
    auto const& products = config.getParameter<std::vector<edm::ParameterSet>>("eventProducts");
    eventProducts_.reserve(products.size());

    for (auto const& product : products) {
      auto const& type = product.getParameter<std::string>("type");
      auto const& label = product.getParameter<std::string>("label");
      auto const& instance = product.getParameter<std::string>("instance");

      // Get the *device* serialiser for the device equivalent of the
      // human-readable host type that we got from the python config. The device
      // serialiser is required to register the H->D transform of each product.
      std::unique_ptr<ngt::SerialiserBase> deviceSerialiser{ngt::SerialiserFactoryDevice::get()->tryToCreate(type)};
      if (!deviceSerialiser) {
        throw edm::Exception(edm::errors::Configuration)
            << "No device serialiser found for host type '" << type
            << "'. A matching device serialiser is required to register the H -> D transformation for this type. "
               "Please ensure that this device serialiser has been registered via "
               "DEFINE_TRIVIAL_SERIALISER_PLUGIN_HOST_DEVICE in the corresponding DataFormats package.";
      }

      Entry entry;

      // "type" comes from the config; it is just the string we use to get the
      // right host serialiser.
      entry.typeName = type;

      // "entry.typeID" is assigned to the actual host product type
      // corresponding to the device type "T" that Serialiser<T> handles.
      entry.typeID = edm::TypeID{deviceSerialiser->hostProductTypeID()};

      entry.getToken = this->consumes(edm::TypeToGet{entry.typeID, edm::PRODUCT_TYPE}, edm::InputTag{label, instance});

      if (deviceSerialiser->hasCopyToDevice()) {
        entry.putToken = this->produces(instance).produces(edm::TypeID{deviceSerialiser->productTypeID()},
                                                           edm::TypeID{deviceSerialiser->hostProductTypeID()},
                                                           deviceSerialiser->preTransformHtoD(),
                                                           deviceSerialiser->transformHtoD());

      } else {
        entry.putToken = this->producesCollector().template produces<edm::Transition::Event>(entry.typeID, instance);
      }

      std::unique_ptr<::ngt::SerialiserBase> hostSerialiser{::ngt::SerialiserFactory::get()->tryToCreate(type)};
      if (!hostSerialiser) {
        throw edm::Exception(edm::errors::Configuration)
            << "No host serialiser found for type '" << type
            << "'. Please ensure the host serialiser has been registered via "
               "DEFINE_TRIVIAL_SERIALISER_PLUGIN in the corresponding DataFormats package.";
      }

      // Cache the serialiser so it does not need to be re-created
      // event-by-event in produce
      entry.serialiser = std::move(hostSerialiser);

      if (verbose_) {
        edm::LogInfo("GenericClonerHost") << "will clone host product of type '" << type << "', label '" << label
                                          << "', instance '" << instance << "'";
      }

      eventProducts_.emplace_back(std::move(entry));
    }
  }

  void GenericClonerHost::produce(edm::Event& event, edm::EventSetup const& /*unused*/) {
    for (auto& entry : eventProducts_) {
      // Get the product from the Event, as a WrapperBase pointer.
      edm::Handle<edm::WrapperBase> handle(entry.typeID.typeInfo());
      event.getByToken(entry.getToken, handle);
      edm::WrapperBase const* wrapper = handle.product();
      if (wrapper == nullptr) {
        throw edm::Exception(edm::errors::ProductNotFound)
            << "Host product of type '" << entry.typeName << "' not found in event.";
      }

      auto reader = entry.serialiser->reader(*wrapper);
      auto writer = entry.serialiser->writer();

      writer->initialize(reader->parameters());

      auto targets = writer->regions();
      auto sources = reader->regions();

      assert(sources.size() == targets.size());
      for (size_t j = 0; j < sources.size(); ++j) {
        assert(sources[j].data() != nullptr);
        assert(targets[j].data() != nullptr);
        assert(targets[j].size_bytes() == sources[j].size_bytes());
        std::memcpy(targets[j].data(), sources[j].data(), sources[j].size_bytes());
      }

      writer->finalize();

      event.put(entry.putToken, writer->get());
    }

    this->putBackend(event);
  }

  void GenericClonerHost::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    descriptions.setComment(
        "This Alpaka EDProducer will clone all the host event products declared by its configuration, "
        "using the host TrivialSerialisation mechanism, and will register at runtime the H -> D transformation "
        "for these products.");

    edm::ParameterSetDescription product;
    product.add<std::string>("type")->setComment(
        "Type name of the host product to be cloned, using the human-readable type alias "
        "(e.g. \"portabletest::TestHostCollection\").");
    product.add<std::string>("label")->setComment("Module label of the producer.");
    product.add<std::string>("instance", "")->setComment("Product instance name.");

    edm::ParameterSetDescription desc;
    desc.addVPSet("eventProducts", product, {})->setComment("Host products to be cloned.");
    desc.addUntracked<bool>("verbose", false)->setComment("Print the type names of the products that will be cloned.");

    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

DEFINE_FWK_ALPAKA_MODULE(ngt::GenericClonerHost);
