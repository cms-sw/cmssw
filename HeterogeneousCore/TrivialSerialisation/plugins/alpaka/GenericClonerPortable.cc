/*
 * This Alpaka EDProducer clones host or device event products declared in
 * its configuration, using the plugin-based NGT trivial serialisation.
 *
 * - Host type aliases (e.g. "portabletest::TestHostCollection") are cloned
 *   using the host TrivialSerialisation mechanism with std::memcpy. If a
 *   matching device serialiser is registered, the H->D transformation is
 *   also registered at construction time.
 *
 * - Device type aliases (e.g. "sistrip::SiStripClusterDevice") are cloned on
 *   device using alpaka::memcpy. The D->H transformation is registered if
 *   available.
 *
 * Products are configured as a VPSet with type and InputTag.
 */

// C++ include files
#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <TBufferFile.h>
#include <TClass.h>

// CMSSW include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/WrapperBaseHandle.h"
#include "FWCore/Framework/interface/WrapperBaseOrphanHandle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ProducerBase.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/ReaderBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserFactory.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/WriterBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/ReaderBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/WriterBase.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt {

  class GenericClonerPortable : public ProducerBase<edm::stream::EDProducer> {
  public:
    explicit GenericClonerPortable(edm::ParameterSet const& config);
    ~GenericClonerPortable() override = default;

    void produce(edm::Event& event, edm::EventSetup const&) final;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    struct Entry {
      std::string typeName;  // human-readable type name from config
      edm::TypeID typeID;
      edm::EDGetToken getToken;
      edm::EDPutToken putToken;
      std::unique_ptr<ngt::SerialiserBase> deviceSerialiser;
      std::unique_ptr<::ngt::SerialiserBase> hostSerialiser;
      edm::TypeWithDict wrappedType;
    };

    std::vector<Entry> eventProducts_;
    bool hasDeviceProducts_ = false;
    bool verbose_;
  };

  GenericClonerPortable::GenericClonerPortable(edm::ParameterSet const& config)
      : ProducerBase<edm::stream::EDProducer>(config), verbose_(config.getUntrackedParameter<bool>("verbose")) {
    auto const& products = config.getParameter<std::vector<edm::ParameterSet>>("products");
    eventProducts_.reserve(products.size());

    for (auto const& product : products) {
      auto const& type = product.getParameter<std::string>("type");
      auto const& src = product.getParameter<edm::InputTag>("src");

      Entry entry;
      entry.typeName = type;

      // Lookup the right serialiser. In order of preference:
      // SerialiserFactoryDevice, SerialiserFactory, ROOT Serialisation.
      //
      // A type alias prefixed with the literal placeholder
      // "ALPAKA_ACCELERATOR_NAMESPACE::" declares (in the config itself)
      // that this product is device-resident; the placeholder is
      // substituted below with this backend's actual namespace. Types
      // without the placeholder skip Check 1 entirely and are resolved as
      // host or ROOT types.
      static std::string const kAlpakaNamespacePlaceholder = "ALPAKA_ACCELERATOR_NAMESPACE::";
      bool const isDeviceType = type.compare(0, kAlpakaNamespacePlaceholder.size(), kAlpakaNamespacePlaceholder) == 0;
      std::string const bareType = isDeviceType ? type.substr(kAlpakaNamespacePlaceholder.size()) : type;

      std::unique_ptr<ngt::SerialiserBase> deviceSerialiser;
      if (isDeviceType) {
        // Check 1: Construct the mangled typeid of the device type from the
        // bare alias (e.g. "sistrip::SiStripClusterDevice"), and use this
        // mangled type to look up a device serialiser.
        std::string const deviceTypeName = std::string(EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)) + "::" + bareType;
        edm::TypeWithDict const deviceTypeTwd = edm::TypeWithDict::byName(deviceTypeName);
        if (bool(deviceTypeTwd)) {
          deviceSerialiser = ngt::SerialiserFactoryDevice::get()->tryToCreate(deviceTypeTwd.typeInfo().name());
        }
        if (!deviceSerialiser) {
          // No ROOT dictionary for this alpaka type alias (e.g. on the
          // CPU/serial backend, where it is just a `using` alias for the
          // host type). Fall back to the namespaced-string key registered by
          // DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN.
          deviceSerialiser = ngt::SerialiserFactoryDevice::get()->tryToCreate(deviceTypeName);
        }
      }

      if (deviceSerialiser) {
        entry.typeID = edm::TypeID{deviceSerialiser->productTypeID()};
        entry.getToken = this->consumes(edm::TypeToGet{entry.typeID, edm::PRODUCT_TYPE}, src);
        hasDeviceProducts_ = true;

        if (deviceSerialiser->hasCopyToHost()) {
          entry.putToken = this->produces(src.instance())
                               .deviceProduces(edm::TypeID{deviceSerialiser->productTypeID()},
                                               edm::TypeID{deviceSerialiser->hostProductTypeID()},
                                               deviceSerialiser->getQueue(),
                                               deviceSerialiser->preTransformDtoH(),
                                               deviceSerialiser->transformDtoH());
        } else {
          entry.putToken =
              this->producesCollector().template produces<edm::Transition::Event>(entry.typeID, src.instance());
        }

        entry.deviceSerialiser = std::move(deviceSerialiser);

        if (verbose_) {
          edm::LogInfo("GenericClonerPortable") << "will clone device product of type '" << type << "', " << src;
        }

        eventProducts_.emplace_back(std::move(entry));
        continue;
      }

      // Check 2: "type" could be a host type alias "T" for which a host
      // serialiser (and perhaps a portable serialiser for the H->D transform)
      // exists.
      edm::TypeWithDict twd = edm::TypeWithDict::byName(bareType);
      std::unique_ptr<ngt::SerialiserBase> portableSerialiser;
      std::unique_ptr<::ngt::SerialiserBase> hostSerialiser;
      if (bool(twd)) {
        portableSerialiser = ngt::SerialiserFactoryDevice::get()->tryToCreate(twd.typeInfo().name());
        hostSerialiser = ::ngt::SerialiserFactory::get()->tryToCreate(twd.typeInfo().name());
      }

      if (hostSerialiser && bool(twd)) {
        entry.typeID = edm::TypeID{twd.typeInfo()};
        entry.getToken = this->consumes(edm::TypeToGet{entry.typeID, edm::PRODUCT_TYPE}, src);

        if (portableSerialiser && portableSerialiser->hasCopyToDevice()) {
          entry.putToken = this->produces(src.instance())
                               .produces(edm::TypeID{portableSerialiser->hostProductTypeID()},
                                         edm::TypeID{portableSerialiser->productTypeID()},
                                         portableSerialiser->preTransformHtoD(),
                                         portableSerialiser->transformHtoD());
        } else {
          entry.putToken =
              this->producesCollector().template produces<edm::Transition::Event>(entry.typeID, src.instance());
        }

        entry.hostSerialiser = std::move(hostSerialiser);

        if (verbose_) {
          edm::LogInfo("GenericClonerPortable") << "will clone host product of type '" << type << "', label '"
                                                << src.label() << "', instance '" << src.instance() << "'";
        }

        eventProducts_.emplace_back(std::move(entry));
        continue;
      }

      // Check 3: Fall back to ROOT serialisation, if a ROOT dictionary is
      // found for this type.
      edm::TypeWithDict wrappedTwd = edm::TypeWithDict::byName("edm::Wrapper<" + bareType + ">");
      if (!twd || !wrappedTwd.getClass()) {
        throw cms::Exception("GenericClonerPortable")
            << "No serialisation mechanism (device or host TrivialSerialisation, or ROOT dictionaries) found for "
               "type '"
            << type
            << "'. Please register a serialiser via DEFINE_TRIVIAL_SERIALISER_PLUGIN or "
               "DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN, or ensure a ROOT dictionary exists for this type.";
      }

      entry.typeID = edm::TypeID{twd.typeInfo()};
      entry.getToken = this->consumes(edm::TypeToGet{entry.typeID, edm::PRODUCT_TYPE}, src);
      entry.putToken =
          this->producesCollector().template produces<edm::Transition::Event>(entry.typeID, src.instance());
      entry.wrappedType = wrappedTwd;

      if (verbose_) {
        edm::LogInfo("GenericClonerPortable") << "will clone ROOT-serialised product of type '" << type << "', label '"
                                              << src.label() << "', instance '" << src.instance() << "'";
      }

      eventProducts_.emplace_back(std::move(entry));
      continue;
    }
  }

  void GenericClonerPortable::produce(edm::Event& event, edm::EventSetup const& /*unused*/) {
    std::unique_ptr<::ALPAKA_ACCELERATOR_NAMESPACE::detail::EDMetadataSentry> sentry;
    if (hasDeviceProducts_) {
      sentry = std::make_unique<::ALPAKA_ACCELERATOR_NAMESPACE::detail::EDMetadataSentry>(event.streamID(),
                                                                                          this->synchronize());
    }

    for (auto& entry : eventProducts_) {
      edm::Handle<edm::WrapperBase> handle(entry.typeID.typeInfo());
      event.getByToken(entry.getToken, handle);
      edm::WrapperBase const* wrapper = handle.product();
      if (wrapper == nullptr) {
        throw edm::Exception(edm::errors::ProductNotFound)
            << "Product of type '" << entry.typeName << "' not found in event.";
      }

      if (entry.hostSerialiser) {
        auto reader = entry.hostSerialiser->reader(*wrapper);
        auto writer = entry.hostSerialiser->writer();

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
      } else if (entry.deviceSerialiser) {
        auto reader = entry.deviceSerialiser->reader(*wrapper, *sentry->metadata());
        auto writer = entry.deviceSerialiser->writer();

        writer->initialize(sentry->metadata()->queue(), reader->parameters());

        auto targets = writer->regions();
        auto sources = reader->regions();

        assert(sources.size() == targets.size());
        for (size_t j = 0; j < sources.size(); ++j) {
          assert(sources[j].data() != nullptr);
          assert(targets[j].data() != nullptr);
          assert(targets[j].size_bytes() == sources[j].size_bytes());
          alpaka::memcpy(sentry->metadata()->queue(),
                         cms::alpakatools::make_device_view(sentry->metadata()->queue(), targets[j]),
                         cms::alpakatools::make_device_view(sentry->metadata()->queue(), sources[j]));
        }

        writer->finalize();
        event.put(entry.putToken, writer->get(sentry->metadata()));
      } else {
        TClass* cls = entry.wrappedType.getClass();
        if (!cls) {
          throw edm::Exception(edm::errors::LogicError)
              << "Failed to get ROOT dictionary class for type '" << entry.typeName << "'.";
        }

        TBufferFile serializedBuffer(TBuffer::kWrite);
        serializedBuffer.WriteObjectAny(wrapper, cls, false);

        serializedBuffer.SetReadMode();
        serializedBuffer.Reset();

        auto clone =
            std::unique_ptr<edm::WrapperBase>(reinterpret_cast<edm::WrapperBase*>(serializedBuffer.ReadObjectAny(cls)));
        if (!clone) {
          throw edm::Exception(edm::errors::LogicError)
              << "Failed to deserialize ROOT product for type '" << entry.typeName << "'.";
        }
        event.put(entry.putToken, std::move(clone));
      }
    }

    this->putBackend(event);
    if (sentry) {
      sentry->finish(true);
    }
  }

  void GenericClonerPortable::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    descriptions.setComment(
        "This Alpaka EDProducer will clone all the host or device event products declared by its configuration, "
        "using the Host or Device TrivialSerialisation mechanism. ");

    edm::ParameterSetDescription product;
    product.add<std::string>("type")->setComment(
        "Type alias of the product to clone. Use the host type alias "
        "(e.g. \"portabletest::TestHostCollection\") to clone a host product, or prefix the device type alias "
        "with the literal \"ALPAKA_ACCELERATOR_NAMESPACE::\" "
        "(e.g. \"ALPAKA_ACCELERATOR_NAMESPACE::sistrip::SiStripClusterDevice\") to clone a device product; "
        "the placeholder is substituted with this backend's actual namespace at construction time.");
    product.add<edm::InputTag>("src")->setComment("InputTag (label and instance) of the product to clone.");

    edm::ParameterSetDescription desc;
    desc.addVPSet("products", product, {})->setComment("Host or device products to be cloned.");
    desc.addUntracked<bool>("verbose", false)->setComment("Print the type names of the products that will be cloned.");

    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

DEFINE_FWK_ALPAKA_MODULE(ngt::GenericClonerPortable);
