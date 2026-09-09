// C++ include files
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <TBufferFile.h>
#include <TClass.h>
#include <TDictAttributeMap.h>

// CMSSW include files
#include "DataFormats/AlpakaCommon/interface/alpaka/EDMetadata.h"
#include "DataFormats/Common/interface/PathStateToken.h"
#include "FWCore/Concurrency/interface/Async.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/WrapperBaseOrphanHandle.h"
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
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ProducerBase.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/chooseDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/MPICore/interface/MPIChannel.h"
#include "HeterogeneousCore/MPICore/interface/MPIToken.h"
#include "HeterogeneousCore/MPICore/interface/MutableOnceFlag.h"
#include "HeterogeneousCore/MPIServices/interface/MPIConsistencyChecker.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserFactory.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/WriterBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserFactoryDevice.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/WriterBase.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // Inherit from ProducerBase. This is so we have access to the EDMetadata,
  // which we need for synchronization
  class MPIReceiverPortable
      : public ProducerBase<edm::stream::EDProducer, edm::ExternalWork, edm::GlobalCache<MutableOnceFlag>> {
  public:
    MPIReceiverPortable(edm::ParameterSet const& config, MutableOnceFlag const* cache)
        : ProducerBase<edm::stream::EDProducer, edm::ExternalWork, edm::GlobalCache<MutableOnceFlag>>(config),
          upstream_(consumes<MPIToken>(config.getParameter<edm::InputTag>("upstream"))),
          token_(this->producesCollector().template produces<MPIToken>()),
          instance_(config.getParameter<int32_t>("instance")) {
      // instance 0 is reserved for the MPIController / MPISource pair instance
      // values greater than 255 may not fit in the MPI tag
      if (instance_ < 1 or instance_ > 255) {
        throw cms::Exception("InvalidValue")
            << "Invalid MPIReceiverPortable instance value, please use a value between 1 and 255";
      }

      auto const& products = config.getParameter<std::vector<edm::ParameterSet>>("products");
      products_.reserve(products.size());
      for (auto const& product : products) {
        auto const& type = product.getParameter<std::string>("type");
        auto const& src = product.getParameter<edm::InputTag>("src");

        // Construct the instance that will be put into the event together with
        // this product, and that will be used by downstream modules to consume
        // this product.
        //
        // edmMpiSplitConfig convention = "src.label@src.instance" if both are
        // set, "label" if only label is set and "instance" if only instance is
        // set
        std::string produceInstance;
        if (src.label().empty()) {
          produceInstance = src.instance();
        } else if (src.instance().empty()) {
          produceInstance = src.label();
        } else {
          produceInstance = src.label() + "@" + src.instance();
        }

        Entry entry;
        entry.typeName = type;

        // Produce PathStateToken but do not transfer it over MPI; the path
        // status is propagated through productCount (set to -1 if the path is
        // inactive).
        if (type == "edm::PathStateToken") {
          entry.token = this->producesCollector().template produces<edm::PathStateToken>();
          products_.emplace_back(std::move(entry));
          continue;
        }

        // Lookup the right serialiser. In order of preference:
        // SerialiserFactoryDevice, SerialiserFactory, ROOT Serialisation.
        //
        // If type contains "ALPAKA_ACCELERATOR_NAMESPACE::", then this product
        // is a device product. Types without this placeholder are resolved as
        // host or ROOT types.
        static std::string const kAlpakaNamespacePlaceholder = "ALPAKA_ACCELERATOR_NAMESPACE::";
        static std::string const kAlpakaNamespace = std::string(EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE)) + "::";

        // Name with "ALPAKA_ACCELERATOR_NAMESPACE::" replaced by this backend's
        // namespace. This will be used to look up a device serialiser.
        std::string deviceTypeName = type;

        bool isDeviceType = false;
        for (size_t pos = deviceTypeName.find(kAlpakaNamespacePlaceholder); pos != std::string::npos;
             pos = deviceTypeName.find(kAlpakaNamespacePlaceholder, pos + kAlpakaNamespace.size())) {
          deviceTypeName.replace(pos, kAlpakaNamespacePlaceholder.size(), kAlpakaNamespace);
          isDeviceType = true;
        }

        // Name with "ALPAKA_ACCELERATOR_NAMESPACE::" removed. This will be used
        // to look up a host serialiser or a ROOT dictionary.
        std::string bareType = type;

        for (size_t pos = bareType.find(kAlpakaNamespacePlaceholder); pos != std::string::npos;
             pos = bareType.find(kAlpakaNamespacePlaceholder, pos)) {
          bareType.erase(pos, kAlpakaNamespacePlaceholder.size());
        }

        std::unique_ptr<ngt::SerialiserBase> deviceSerialiser;
        if (isDeviceType) {
          // Check 1: Use the mangled typeid of the resolved device type (e.g.
          // "alpaka_cuda_async::sistrip::SiStripClusterDevice") to look up a
          // device serialiser.
          LogDebug("MPIReceiverPortable") << "looking for device serialiser for type \"" << type << "\"";
          edm::TypeWithDict const deviceTypeTwd = edm::TypeWithDict::byName(deviceTypeName);
          if (bool(deviceTypeTwd)) {
            deviceSerialiser = ngt::SerialiserFactoryDevice::get()->tryToCreate(deviceTypeTwd.typeInfo().name());
          }
          if (!deviceSerialiser) {
            // Try directly with the type name
            deviceSerialiser = ngt::SerialiserFactoryDevice::get()->tryToCreate(deviceTypeName);
          }
        }

        if (deviceSerialiser) {
          edm::TypeID typeID{deviceSerialiser->productTypeID()};
          hasDeviceProducts_ = true;

          LogDebug("MPIReceiverPortable") << "found device serialiser for type \"" << type << "\"";

          if (deviceSerialiser->hasCopyToHost()) {
            LogDebug("MPIReceiverPortable") << "Registering D to H transform for type \"" << type << "\"";
            // Register the D to H transform
            entry.token = this->produces(produceInstance)
                              .deviceProduces(edm::TypeID{deviceSerialiser->productTypeID()},
                                              edm::TypeID{deviceSerialiser->hostProductTypeID()},
                                              deviceSerialiser->getQueue(),
                                              deviceSerialiser->preTransformDtoH(),
                                              deviceSerialiser->transformDtoH());
          } else {
            LogDebug("MPIReceiverPortable") << "No D to H transform found for type \"" << type << "\"";
            entry.token = this->producesCollector().template produces<edm::Transition::Event>(typeID, produceInstance);
          }
          entry.deviceSerialiser = std::move(deviceSerialiser);

          LogDebug("MPIReceiverPortable") << "receive device type \"" << typeID << "\" (" << type << ") for instance \""
                                          << produceInstance << "\" over MPI channel instance " << instance_;

          products_.emplace_back(std::move(entry));
          continue;
        }

        // Check 2: "type" could be a host type alias "T" for which a host
        // serialiser (and perhaps a portable serialiser for the H->D transform)
        // exists.
        edm::TypeWithDict twd = edm::TypeWithDict::byName(bareType);
        std::unique_ptr<ngt::SerialiserBase> portableSerialiser;
        std::unique_ptr<::ngt::SerialiserBase> hostSerialiser;
        LogDebug("MPIReceiverPortable") << "looking for host serialiser for type \"" << type << "\"";
        if (bool(twd)) {
          portableSerialiser = ngt::SerialiserFactoryDevice::get()->tryToCreate(twd.typeInfo().name());
          hostSerialiser = ::ngt::SerialiserFactory::get()->tryToCreate(twd.typeInfo().name());
        }

        if (hostSerialiser && bool(twd)) {
          edm::TypeID typeID{twd.typeInfo()};
          LogDebug("MPIReceiverPortable") << "found host serialiser for type \"" << type << "\"";

          if (portableSerialiser && portableSerialiser->hasCopyToDevice()) {
            LogDebug("MPIReceiverPortable") << "Registering H to D transform for type \"" << type << "\"";
            // Register the H to D transform
            entry.token = this->produces(produceInstance)
                              .produces(edm::TypeID{portableSerialiser->hostProductTypeID()},
                                        edm::TypeID{portableSerialiser->productTypeID()},
                                        portableSerialiser->preTransformHtoD(),
                                        portableSerialiser->transformHtoD());
          } else {
            LogDebug("MPIReceiverPortable") << "No H to D transform found for type \"" << type << "\"";
            entry.token = this->producesCollector().template produces<edm::Transition::Event>(typeID, produceInstance);
          }
          entry.hostSerialiser = std::move(hostSerialiser);

          LogDebug("MPIReceiverPortable") << "receive host type \"" << typeID << "\" (" << type << ") for instance \""
                                          << produceInstance << "\" over MPI channel instance " << instance_;

          products_.emplace_back(std::move(entry));
          continue;
        }

        // Check 3: Fall back to ROOT serialisation, if a ROOT dictionary is
        // found for this type
        edm::TypeWithDict wrappedTwd = edm::TypeWithDict::byName("edm::Wrapper<" + bareType + ">");
        LogDebug("MPIReceiverPortable") << "looking for ROOT serialisation of type \"" << type
                                        << "\" (wrapper resolved: " << wrappedTwd.typeInfo().name() << ")";
        if (!twd || !wrappedTwd.getClass()) {
          throw cms::Exception("MPIReceiverPortable")
              << "No serialisation mechanism (device or host TrivialSerialisation, or ROOT dictionaries) found for "
                 "type '"
              << type
              << "'. Either register a serialiser via DEFINE_TRIVIAL_SERIALISER_PLUGIN or "
                 "DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN, or make sure a ROOT dictionary exists for this type.";
        }
        // Check if the type is transient. If it is, it cannot be serialised
        // with ROOT.
        TDictAttributeMap* attributes = wrappedTwd.getClass()->GetAttributeMap();
        if (attributes and attributes->HasKey("persistent") and
            std::strcmp(attributes->GetPropertyAsString("persistent"), "false") == 0) {
          // persistent = false -> transient = true
          throw cms::Exception("MPIReceiverPortable")
              << "Type '" << type
              << "' is transient (persistent = \"false\"), so it cannot be serialised with ROOT. "
                 "Please register a serialiser via DEFINE_TRIVIAL_SERIALISER_PLUGIN or "
                 "DEFINE_TRIVIAL_SERIALISER_PORTABLE_PLUGIN.";
        }
        edm::TypeID typeID{twd.typeInfo()};
        entry.token = this->producesCollector().template produces<edm::Transition::Event>(typeID, produceInstance);
        entry.wrappedType = wrappedTwd;

        LogDebug("MPIReceiverPortable") << "found ROOT dictionary for type \"" << type << "\"";
        LogDebug("MPIReceiverPortable") << "receive ROOT type \"" << typeID << "\" (" << type << ") for instance \""
                                        << produceInstance << "\" over MPI channel instance " << instance_;

        products_.emplace_back(std::move(entry));
      }

      // record information about this receiver for configuration consistency check
      edm::Service<MPIConsistencyChecker> module_info_service;
      std::vector<std::string> product_types;
      product_types.reserve(products_.size());
      for (auto const& entry : products_) {
        product_types.push_back(entry.typeName);
      }
      std::string module_label = config.getParameter<std::string>("@module_label");
      std::string upstream_label = config.getParameter<edm::InputTag>("upstream").label();
      if (cache == nullptr) {
        throw cms::Exception("MPIReceiverPortable") << "MPIReceiverPortable's global cache is null";
      }
      std::call_once(cache->information_recorded_flag, [&]() {
        module_info_service->recordMPIModuleInfo(false, module_label, upstream_label, this->instance_, product_types);
      });
    }

    static std::unique_ptr<MutableOnceFlag> initializeGlobalCache(edm::ParameterSet const&) {
      return std::make_unique<MutableOnceFlag>();
    }

    static void globalEndJob(MutableOnceFlag const*) {}

    void acquire(edm::Event const& event, edm::EventSetup const&, edm::WaitingTaskWithArenaHolder holder) final {
      // reset the metadata that could have been left behind by a previous event
      metadata_.reset();
      if (hasDeviceProducts_) {
        metadata_ = std::make_shared<EDMetadata>(detail::chooseDevice(event.streamID()));
      }

      const MPIToken& token = event.get(upstream_);

      receivedProductMetadata_ = std::make_shared<ProductMetadataBuilder>();
      receivedWrappers_.resize(products_.size());
      asyncWorkLaunched_ = false;

      edm::Service<edm::Async> as;
      as->runAsync(
          std::move(holder),
          [this, token]() {
            token.channel()->receiveMetadata(instance_, receivedProductMetadata_);
#ifdef EDM_ML_DEBUG
            receivedProductMetadata_->debugPrintMetadataSummary();
#endif

            if (receivedProductMetadata_->productCount() == -1) {
              return;
            }

            std::unique_ptr<TBufferFile> serialized_buffer;
            if (receivedProductMetadata_->hasSerialized()) {
              serialized_buffer =
                  token.channel()->receiveSerializedBuffer(instance_, receivedProductMetadata_->serializedBufferSize());
            }

            struct PendingDeviceWriter {
              size_t index;
              std::unique_ptr<ngt::WriterBase> writer;
            };
            struct PendingHostWriter {
              size_t index;
              std::unique_ptr<::ngt::WriterBase> writer;
            };

            std::vector<MPI_Request> requests;
            std::vector<PendingDeviceWriter> pendingDeviceWriters;
            std::vector<PendingHostWriter> pendingHostWriters;

            for (size_t i = 0; i < products_.size(); ++i) {
              auto const& entry = products_[i];

              // PathStateToken is not transferred; it is handled in produce().
              if (entry.typeName == "edm::PathStateToken") {
                continue;
              }

              auto product_meta = receivedProductMetadata_->getNext();

              if (product_meta.kind == ProductMetadata::Kind::Missing) {
                continue;
              }

              if (product_meta.kind == ProductMetadata::Kind::Serialized) {
                if (!serialized_buffer) {
                  throw cms::Exception("MPIReceiverPortable")
                      << "Received a Serialized product kind for '" << entry.typeName
                      << "' but no serialized buffer was received.";
                }
                TClass* cls = entry.wrappedType.getClass();
                if (!cls) {
                  throw cms::Exception("MPIReceiverPortable")
                      << "Failed to get TClass for ROOT product '" << entry.typeName << "'.";
                }
                auto wrapper = std::unique_ptr<edm::WrapperBase>(reinterpret_cast<edm::WrapperBase*>(cls->New()));
                cls->Streamer(wrapper.get(), *serialized_buffer);
                receivedWrappers_[i] = std::move(wrapper);
                continue;
              }

              if (product_meta.kind != ProductMetadata::Kind::TrivialCopy) {
                throw cms::Exception("MPIReceiverPortable")
                    << "Unexpected product metadata kind for product '" << entry.typeName << "'.";
              }

              // At this point, all remaining products should be of type
              // ProductMetadata::Kind::TrivialCopy, and thus a serialiser (host
              // or device) should exist for them.

              if (entry.deviceSerialiser) {
                auto writer = entry.deviceSerialiser->writer();
                ::ngt::AnyBuffer buffer = writer->uninitialized_parameters();
                if (buffer.size_bytes() != product_meta.sizeMeta) {
                  throw cms::Exception("MPIReceiverPortable")
                      << "Buffer size mismatch for device product '" << entry.typeName << "': deviceSerialiser expects "
                      << buffer.size_bytes() << " bytes of metadata, but sender sent " << product_meta.sizeMeta
                      << " bytes.";
                }
                std::memcpy(buffer.data(), product_meta.trivialCopyOffset, product_meta.sizeMeta);

                writer->initialize(metadata_->queue(), buffer);
                asyncWorkLaunched_ = true;
                token.channel()->receiveInitializedTrivialCopyAsync(instance_, *writer, requests);
                pendingDeviceWriters.push_back({i, std::move(writer)});
              } else {
                // Host path: allocate host buffer, then post a non-blocking receive.
                auto writer = entry.hostSerialiser->writer();
                ::ngt::AnyBuffer buffer = writer->uninitialized_parameters();
                if (buffer.size_bytes() != product_meta.sizeMeta) {
                  throw cms::Exception("MPIReceiverPortable")
                      << "Buffer size mismatch for host product '" << entry.typeName << "': Serialiser expects "
                      << buffer.size_bytes() << " bytes of metadata, but sender sent " << product_meta.sizeMeta
                      << " bytes.";
                }
                std::memcpy(buffer.data(), product_meta.trivialCopyOffset, product_meta.sizeMeta);

                writer->initialize(buffer);
                token.channel()->receiveInitializedTrivialCopyAsync(instance_, *writer, requests);
                pendingHostWriters.push_back({i, std::move(writer)});
              }
            }

            // Wait for all non-blocking receives to complete.
            MPIChannel::waitAll(requests);

            for (auto& pending : pendingDeviceWriters) {
              pending.writer->finalize();
              receivedWrappers_[pending.index] = pending.writer->get(metadata_);
            }
            for (auto& pending : pendingHostWriters) {
              pending.writer->finalize();
              receivedWrappers_[pending.index] = pending.writer->get();
            }
          },
          []() { return "Calling MPIReceiverPortable::acquire()"; });
    }

    void produce(edm::Event& event, edm::EventSetup const&) final {
      std::unique_ptr<detail::EDMetadataSentry> sentry;
      if (metadata_) {
        sentry = std::make_unique<detail::EDMetadataSentry>(std::move(metadata_), this->synchronize());
      }

      MPIToken token = event.get(upstream_);

      if (receivedProductMetadata_->productCount() == -1) {
        event.emplace(token_, token);
        this->putBackend(event);
        if (sentry) {
          sentry->finish(false);
        }
        return;
      }

      for (size_t i = 0; i < products_.size(); ++i) {
        auto const& entry = products_[i];

        if (entry.typeName == "edm::PathStateToken") {
          // Put a fresh PathStateToken into the event, since the one created
          // remotely was not transferred.
          event.put(entry.token, std::make_unique<edm::PathStateToken>());
          continue;
        }

        if (!receivedWrappers_[i]) {
          edm::LogWarning("MPIReceiverPortable") << "Product " << entry.typeName << " was not received.";
          continue;
        }

        event.put(entry.token, std::move(receivedWrappers_[i]));
      }

      event.emplace(token_, token);
      this->putBackend(event);
      if (sentry) {
        sentry->finish(asyncWorkLaunched_);
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      descriptions.setComment(
          "This module can receive arbitrary device or host event products from an "
          "\"MPISenderPortable\" module in a separate CMSSW job, and produce them into the event.");

      edm::ParameterSetDescription product;
      product.add<std::string>("type")->setComment(
          "Type alias of the product. Use \"ALPAKA_ACCELERATOR_NAMESPACE::\" wherever the backend namespace "
          "appears in the type (e.g. \"ALPAKA_ACCELERATOR_NAMESPACE::sistrip::SiStripClusterDevice\") to indicate "
          "that it is a device product; every occurrence of the placeholder is substituted with the backend's "
          "actual namespace at construction time. "
          "For host and ROOT products, use the plain C++ type name with no placeholder.");
      product.add<edm::InputTag>("src", edm::InputTag{})->setComment("InputTag identifying the product to produce. ");

      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("upstream", {"source"})
          ->setComment(
              "MPI communication channel. Can be an \"MPIController\", \"MPISource\", or "
              "\"MPISenderPortable\"/\"MPIReceiverPortable\".");
      desc.addVPSet("products", product, {})
          ->setComment("Host or device products to be received from a separate CMSSW job.");
      desc.add<int32_t>("instance", 0)
          ->setComment(
              "A value between 1 and 255 used to identify a matching pair of "
              "\"MPISenderPortable\"/\"MPIReceiverPortable\".");

      descriptions.addWithDefaultLabel(desc);
    }

  private:
    struct Entry {
      std::string typeName;  // type name from config (for PathStateToken check and logging)
      edm::EDPutToken token;
      std::unique_ptr<ngt::SerialiserBase> deviceSerialiser;
      std::unique_ptr<::ngt::SerialiserBase> hostSerialiser;
      edm::TypeWithDict wrappedType;
    };

    edm::EDGetTokenT<MPIToken> const upstream_;
    edm::EDPutTokenT<MPIToken> const token_;
    std::vector<Entry> products_;
    int32_t const instance_;
    bool hasDeviceProducts_ = false;

    std::shared_ptr<ProductMetadataBuilder> receivedProductMetadata_;
    std::vector<std::unique_ptr<edm::WrapperBase>> receivedWrappers_;
    bool asyncWorkLaunched_ = false;
    std::shared_ptr<EDMetadata> metadata_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(MPIReceiverPortable);
