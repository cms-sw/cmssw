#ifndef HeterogeneousCore_AlpakaCore_interface_ProducerBase_h
#define HeterogeneousCore_AlpakaCore_interface_ProducerBase_h

#include "DataFormats/AlpakaCommon/interface/alpaka/DeviceProductType.h"
#include "DataFormats/Common/interface/DeviceProduct.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Framework/interface/moduleAbilities.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadataAcquireSentry.h"
#include "HeterogeneousCore/AlpakaCore/interface/modulePrevalidate.h"
#include "HeterogeneousCore/AlpakaInterface/interface/Backend.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

#include <any>
#include <cassert>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

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
   * backends the deviceProduces() registers the automatic copy to
   * host and a transformation from edm::DeviceProduct<T> to U, where
   * U is the host-equivalent of T. The transformation from T to U is
   * done by a specialization of cms::alpakatools::CopyToHost<T> class
   * template, that should be provided in the same file where T is
   * defined
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
    ProducerBase(edm::ParameterSet const& iConfig)
        : backendToken_(Base::produces("backend")),
          // The 'synchronize' parameter can be unset in Alpaka
          // modules specified with the namespace prefix instead if
          // '@alpaka' suffix
          synchronize_(iConfig.getUntrackedParameter<edm::ParameterSet>("alpaka").getUntrackedParameter<bool>(
              "synchronize", false)) {}

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
      cms::alpakatools::modulePrevalidate(descriptions);
    }

  protected:
    void putBackend(edm::Event& iEvent) const {
      iEvent.emplace(this->backendToken_, static_cast<unsigned short>(kBackend));
    }

    bool synchronize() const { return synchronize_; }

  private:
    edm::EDPutTokenT<unsigned short> const backendToken_;
    bool const synchronize_ = false;

    template <typename TProducer, edm::Transition Tr>
    friend class ProducerBaseAdaptor;

    // Host products
    //
    // intentionally not returning BranchAliasSetter
    // can think of it later if really needed
    template <typename TProduct, edm::Transition Tr>
    edm::EDPutTokenT<TProduct> produces(std::string instanceName) {
      constexpr bool hasCopy = requires(Queue& queue, TProduct const& prod) {
        cms::alpakatools::CopyToDevice<TProduct>::copyAsync(queue, prod);
      };

      if constexpr (detail::useProductDirectly or not hasCopy) {
        return Base::template produces<TProduct, Tr>(std::move(instanceName));
      } else {
        edm::EDPutTokenT<TProduct> hostToken = Base::template produces<TProduct, Tr>(instanceName);
        this->registerTransformAsync(
            hostToken,
            [synchronize = this->synchronize()](
                edm::StreamID streamID, TProduct const& hostProduct, edm::WaitingTaskWithArenaHolder holder) {
              detail::EDMetadataAcquireSentry sentry(streamID, std::move(holder), synchronize);
              using CopyT = cms::alpakatools::CopyToDevice<TProduct>;
              auto productOnDevice = CopyT::copyAsync(sentry.metadata()->queue(), hostProduct);
              // Need to keep the EDMetadata object from sentry.finish()
              // alive until the synchronization
              using TplType = std::tuple<std::shared_ptr<EDMetadata>, decltype(productOnDevice)>;
              // Wrap possibly move-only type into a copyable type
              return std::make_shared<TplType>(sentry.finish(), std::move(productOnDevice));
            },
            [](edm::StreamID, auto tplPtr) {
              using DeviceObject = std::tuple_element_t<1, std::remove_cvref_t<decltype(*tplPtr)>>;
              using DeviceProductType = detail::DeviceProductType<DeviceObject>;
              return DeviceProductType(std::move(std::get<0>(*tplPtr)), std::move(std::get<1>(*tplPtr)));
            },
            std::move(instanceName));
        return hostToken;
      }
    }

    // Type-erased overload of produces() for host products
    //
    // This overload does not check whether a CopyToDevice exists for the
    // produced type. It always registers its host-to-device transformation,
    // without a fallback. The caller must ensure that a HtoD copy operation
    // exists for this type and provide the corresponding copyAsync and
    // transform callables.
    //
    // This overload does not support producing host-only products.
    template <edm::Transition Tr, typename TCopyAsync, typename TTransform>
    edm::EDPutToken produces(edm::TypeID hostProductType,
                             edm::TypeID deviceProductType,
                             std::string instanceName,
                             TCopyAsync&& copyAsync,
                             TTransform&& transform) {
      edm::EDPutToken token = Base::template produces<Tr>(hostProductType, instanceName);

      if constexpr (not detail::useProductDirectly) {
        using TplType = std::tuple<std::any, std::shared_ptr<EDMetadata>>;

        this->registerTransformAsync(
            token,
            [copyAsync = std::forward<TCopyAsync>(copyAsync), synchronize = this->synchronize()](
                edm::StreamID streamID, edm::WrapperBase const& wb, edm::WaitingTaskWithArenaHolder holder) {
              detail::EDMetadataAcquireSentry sentry(streamID, std::move(holder), synchronize);
              auto productOnDevice = copyAsync(sentry.metadata()->queue(), wb);
              return std::make_shared<TplType>(std::move(productOnDevice), sentry.finish());
            },
            [transform = std::forward<TTransform>(transform)](edm::StreamID, std::shared_ptr<TplType> tplPtr) {
              auto& productOnDevice = std::get<0>(*tplPtr);
              return transform(productOnDevice, std::move(std::get<1>(*tplPtr)));
            },
            deviceProductType,
            std::move(instanceName));
      } else {
        assert(hostProductType == deviceProductType &&
               "On the synchronous CPU backend, only types whose CopyToHost<T>::copyAsync returns T itself are "
               "supported.");
      }

      return token;
    }

    // Device products
    //
    // intentionally not returning BranchAliasSetter
    // can think of it later if really needed
    template <typename TProduct, typename TToken, edm::Transition Tr>
    edm::EDPutTokenT<TToken> deviceProduces(std::string instanceName) {
      if constexpr (detail::useProductDirectly) {
        return Base::template produces<TToken, Tr>(std::move(instanceName));
      } else {
        edm::EDPutTokenT<TToken> token = Base::template produces<TToken, Tr>(instanceName);
        using CopyT = cms::alpakatools::CopyToHost<TProduct>;
        this->registerTransformAsync(
            token,
            [](edm::StreamID, TToken const& deviceProduct, edm::WaitingTaskWithArenaHolder holder) {
              // Reuse the queue associated o the device product for the host copy.
              auto queue = deviceProduct.template metadata<EDMetadata>().shared_queue();
              detail::EDMetadataAcquireSentry sentry(queue, std::move(holder));
              auto metadataPtr = sentry.metadata();
              TProduct const& productOnDevice = deviceProduct.template getSynchronized<EDMetadata>(*metadataPtr);

              auto productOnHost = CopyT::copyAsync(*queue, productOnDevice);

              // Need to keep the EDMetadata object from sentry.finish()
              // alive until the synchronization
              using TplType = std::tuple<decltype(productOnHost), std::shared_ptr<EDMetadata>>;
              // Wrap possibly move-only type into a copyable type
              return std::make_shared<TplType>(std::move(productOnHost), sentry.finish());
            },
            [](edm::StreamID, auto tplPtr) {
              auto& productOnHost = std::get<0>(*tplPtr);
              if constexpr (requires { CopyT::postCopy(productOnHost); }) {
                CopyT::postCopy(productOnHost);
              }
              return std::move(productOnHost);
            },
            std::move(instanceName));
        return token;
      }
    }

    template <edm::Transition Tr, typename TGetQueue, typename TCopyAsync, typename TTransform>
    edm::EDPutToken deviceProduces(edm::TypeID deviceProductType,
                                   edm::TypeID hostProductType,
                                   std::string instanceName,
                                   TGetQueue&& getQueue,
                                   TCopyAsync&& copyAsync,
                                   TTransform&& transform) {
      edm::EDPutToken token = Base::template produces<Tr>(deviceProductType, instanceName);

      if constexpr (not detail::useProductDirectly) {
        using TplType = std::tuple<std::any, std::shared_ptr<EDMetadata>>;

        this->registerTransformAsync(
            token,
            [getQueue = std::forward<TGetQueue>(getQueue), copyAsync = std::forward<TCopyAsync>(copyAsync)](
                edm::StreamID, edm::WrapperBase const& wb, edm::WaitingTaskWithArenaHolder holder) {
              auto queue = getQueue(wb);
              detail::EDMetadataAcquireSentry sentry(queue, std::move(holder));
              auto metadataPtr = sentry.metadata();
              auto productOnHost = copyAsync(*queue, *metadataPtr, wb);
              return std::make_shared<TplType>(std::move(productOnHost), sentry.finish());
            },
            [transform = std::forward<TTransform>(transform)](edm::StreamID, std::shared_ptr<TplType> tplPtr) {
              auto productOnHost = std::get<0>(*tplPtr);
              return transform(productOnHost);
            },
            hostProductType,
            std::move(instanceName));
      }

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

    // for runtime-typed host-only products
    template <typename TCopyAsync, typename TTransform>
    edm::EDPutToken produces(edm::TypeID hostProductType,
                             edm::TypeID deviceProductType,
                             TCopyAsync&& copyAsync,
                             TTransform&& transform) {
      return producer_.template produces<Tr>(hostProductType,
                                             deviceProductType,
                                             label_,
                                             std::forward<TCopyAsync>(copyAsync),
                                             std::forward<TTransform>(transform));
    }

    // for device products
    template <typename TProduct, typename TToken>
    edm::EDPutTokenT<TToken> deviceProduces() {
      return producer_.template deviceProduces<TProduct, TToken, Tr>(label_);
    }

    // for runtime-typed device products
    template <typename TGetQueue, typename TCopyAsync, typename TTransform>
    edm::EDPutToken deviceProduces(edm::TypeID deviceProductType,
                                   edm::TypeID hostProductType,
                                   TGetQueue&& getQueue,
                                   TCopyAsync&& copyAsync,
                                   TTransform&& transform) {
      return producer_.template deviceProduces<Tr>(deviceProductType,
                                                   hostProductType,
                                                   label_,
                                                   std::forward<TGetQueue>(getQueue),
                                                   std::forward<TCopyAsync>(copyAsync),
                                                   std::forward<TTransform>(transform));
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
