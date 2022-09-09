#ifndef HeterogeneousCore_AlpakaCore_interface_ProducerBase_h
#define HeterogeneousCore_AlpakaCore_interface_ProducerBase_h

#include "DataFormats/Common/interface/DeviceProduct.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Framework/interface/moduleAbilities.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataAcquireSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/EventCache.h"
#include "HeterogeneousCore/AlpakaCore/interface/QueueCache.h"
#include "HeterogeneousCore/AlpakaCore/interface/module_backend_config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/TransferToHost.h"

#include <memory>
#include <tuple>

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template <typename Producer, edm::Transition Tr>
  class ProducerBaseAdaptor;

  /**
   * The ProducerBase acts as a common base class for all Alpaka
   * EDProducers. The main benefit is to have a single place for the
   * definition of produces() functions.
   *
   * The produces() functions return a custom ProducerBaseAdaptor in
   * order to call the deviceProduces(). For device or asynchronous
   * backends the deviceProduces() registers the automatic transfer to
   * host and a transformation from edm::DeviceProduct<T> to U, where
   * U is the host-equivalent of T. The transformation from T to U is
   * done by a specialization of cms::alpakatools::TransferToHost<T>
   * template, that should be provided in the same file where T is
   * defined.
   *
   * TODO: add "override" for labelsForToken()
   */
  template <template <typename...> class BaseT, typename... Args>
  class ProducerBase : public BaseT<Args..., edm::Transformer> {
    static_assert(not edm::CheckAbility<edm::module::Abilities::kTransformer>::kHasIt,
                  "ALPAKA_ACCELERATOR_NAMESPACE::ProducerBase can not be used with Transformer ability (as it is "
                  "used internally)");
    using Base = BaseT<Args..., edm::Transformer>;

  public:
    template <edm::Transition Tr = edm::Transition::Event>
    [[nodiscard]] auto produces() noexcept {
      return ProducerBaseAdaptor<ProducerBase, Tr>(*this);
    }

    template <edm::Transition Tr = edm::Transition::Event>
    [[nodiscard]] auto produces(std::string instanceName) noexcept {
      return ProducerBaseAdaptor<ProducerBase, Tr>(*this, std::move(instanceName));
    }

    static void prevalidate(edm::ConfigurationDescriptions& descriptions) {
      Base::prevalidate(descriptions);
      cms::alpakatools::module_backend_config(descriptions);
    }

  private:
    template <typename TProducer, edm::Transition Tr>
    friend class ProducerBaseAdaptor;

    // Host products
    //
    // intentionally not returning BranchAliasSetter
    // can think of it later if really needed
    template <typename TProduct, edm::Transition Tr>
    edm::EDPutTokenT<TProduct> produces(std::string instanceName) {
      return Base::template produces<TProduct, Tr>(std::move(instanceName));
    }

    // Device products
    //
    // intentionally not returning BranchAliasSetter
    // can think of it later if really needed
    template <typename TProduct, typename TToken, edm::Transition Tr>
    edm::EDPutTokenT<TToken> deviceProduces(std::string instanceName) {
      edm::EDPutTokenT<TToken> token = Base::template produces<TToken, Tr>(std::move(instanceName));
      // TODO: should host-synchronous backend(s) have DeviceProduct<TProduct> branch?
#ifndef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
      // All device backends or devices with asynchronous queue

      // TODO: in principle we could also decide to not register the
      // transfer to host if the TransferToHost<T> has not been
      // specialized for TProduct.
      this->registerTransformAsync(
          token,
          [](TToken const& deviceProduct, edm::WaitingTaskWithArenaHolder holder) {
            auto const& device = alpaka::getDev(deviceProduct.template metadata<EDMetadata>().queue());
            detail::EDMetadataAcquireSentry sentry(device, std::move(holder));
            auto metadataPtr = sentry.metadata();
            constexpr bool tryReuseQueue = true;
            TProduct const& productOnDevice =
                deviceProduct.template getSynchronized<EDMetadata>(*metadataPtr, tryReuseQueue);

            using TransferT = cms::alpakatools::TransferToHost<TProduct>;
            using HostType = typename TransferT::HostDataType;
            HostType productOnHost = TransferT::transferAsync(metadataPtr->queue(), productOnDevice);

            // Need to keep the EDMetadata object from sentry.finish()
            // alive until the synchronization
            using TplType = std::tuple<HostType, std::shared_ptr<EDMetadata>>;
            // Wrap possibly move-only type into a copyable type
            return std::make_shared<TplType>(std::move(productOnHost), sentry.finish());
          },
          [](auto tplPtr) { return std::move(std::get<0>(*tplPtr)); });
#endif
      return token;
    }
  };

  // Adaptor class to make the type-deducing produces() calls to work
  template <typename TProducer, edm::Transition Tr>
  class ProducerBaseAdaptor {
  public:
    // for host-only products
    template <typename Type>
    edm::EDPutTokenT<Type> produces() {
      return producer_.template produces<Type, Tr>(label_);
    }

    // for device products
    template <typename TProduct, typename TToken>
    edm::EDPutTokenT<TToken> deviceProduces() {
      return producer_.template deviceProduces<TProduct, TToken, Tr>(label_);
    }

  private:
    // only ProducerBase is allowed to make an instance of this class
    friend TProducer;

    ProducerBaseAdaptor(TProducer& producer, std::string label) : producer_(producer), label_(std::move(label)) {}
    explicit ProducerBaseAdaptor(TProducer& producer) : producer_(producer) {}

    TProducer& producer_;
    std::string const label_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
