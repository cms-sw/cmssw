#ifndef HeterogeneousCore_TrivialSerialisation_interface_alpaka_Serialiser_h
#define HeterogeneousCore_TrivialSerialisation_interface_alpaka_Serialiser_h

#include <any>
#include <memory>
#include <typeinfo>

#include "DataFormats/AlpakaCommon/interface/alpaka/DeviceProductType.h"
#include "DataFormats/AlpakaCommon/interface/alpaka/EDMetadata.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/Reader.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/Writer.h"

namespace ngt::detail {
  [[noreturn]] void throwHostProductTypeIDError();
}

namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt {

  template <typename T, typename TQueue>
  concept HasCopyToHost = requires(TQueue& q, T const& t) {
    { cms::alpakatools::CopyToHost<T>::copyAsync(q, t) };
  };

  template <typename T, typename TQueue>
  concept HasCopyToDevice = requires(TQueue& q, T const& t) {
    { cms::alpakatools::CopyToDevice<T>::copyAsync(q, t) };
  };

  // Get through CopyToHost<T> the host-equivalent of T
  template <typename T, typename TQueue>
    requires HasCopyToHost<T, TQueue>
  using HostTypeOf = std::remove_cvref_t<decltype(cms::alpakatools::CopyToHost<T>::copyAsync(
      std::declval<TQueue&>(), std::declval<T const&>()))>;

  // Concrete Serialiser for device products.
  // T is the inner product type (e.g. PortableDeviceCollection<...>).
  template <typename T>
  class Serialiser : public SerialiserBase {
  public:
    using WrapperType = edm::Wrapper<detail::DeviceProductType<T>>;

    std::unique_ptr<WriterBase> writer() override { return std::make_unique<Writer<T>>(); }

    std::unique_ptr<const ReaderBase> reader(edm::WrapperBase const& wrapper, EDMetadata& metadata) override {
      WrapperType const& w = dynamic_cast<WrapperType const&>(wrapper);
      if constexpr (detail::useProductDirectly) {
        return std::make_unique<Reader<T>>(w.bareProduct());
      } else {
        return std::make_unique<Reader<T>>(w.bareProduct().template getSynchronized<EDMetadata>(metadata));
      }
    }

    // The methods below are not really related to serialisation. They might be
    // moved elsewhere when a better place for them is found.
    std::type_info const& productTypeID() const override { return typeid(detail::DeviceProductType<T>); }

    std::type_info const& hostProductTypeID() const override {
      if constexpr (HasCopyToHost<T, Queue>) {
        return typeid(HostTypeOf<T, Queue>);
      } else {
        ::ngt::detail::throwHostProductTypeIDError();
      }
    }

    bool hasCopyToHost() const override { return HasCopyToHost<T, Queue>; }

    bool hasCopyToDevice() const override {
      if constexpr (HasCopyToHost<T, Queue>) {
        return HasCopyToDevice<HostTypeOf<T, Queue>, Queue>;
      } else {
        return false;
      }
    }

    std::function<std::shared_ptr<Queue>(edm::WrapperBase const&)> getQueue() const override {
      if constexpr (not detail::useProductDirectly) {
        return [](edm::WrapperBase const& wb) -> std::shared_ptr<Queue> {
          auto const& deviceProduct = dynamic_cast<WrapperType const&>(wb).bareProduct();
          return deviceProduct.template metadata<EDMetadata>().shared_queue();
        };
      }
      return nullptr;
    }

    std::function<std::any(Queue&, EDMetadata&, edm::WrapperBase const&)> preTransformDtoH() const override {
      if constexpr (HasCopyToHost<T, Queue>) {
        using CopyT = cms::alpakatools::CopyToHost<T>;
        using HostProductType = HostTypeOf<T, Queue>;

        return [](Queue& queue, EDMetadata& metadata, edm::WrapperBase const& wb) -> std::any {
          auto const& deviceProduct = dynamic_cast<WrapperType const&>(wb).bareProduct();
          T const& productOnDevice = deviceProduct.template getSynchronized<EDMetadata>(metadata);
          auto productOnHost = CopyT::copyAsync(queue, productOnDevice);
          return std::make_shared<HostProductType>(std::move(productOnHost));
        };
      } else {
        return nullptr;
      }
    }

    std::function<std::unique_ptr<edm::WrapperBase>(std::any const&)> transformDtoH() const override {
      if constexpr (HasCopyToHost<T, Queue>) {
        using CopyT = cms::alpakatools::CopyToHost<T>;
        using HostProductType = HostTypeOf<T, Queue>;

        return [](std::any const& cache) -> std::unique_ptr<edm::WrapperBase> {
          auto& productOnHost = *std::any_cast<std::shared_ptr<HostProductType>>(cache);
          if constexpr (requires { CopyT::postCopy(productOnHost); }) {
            CopyT::postCopy(productOnHost);
          }
          return std::make_unique<edm::Wrapper<HostProductType>>(edm::WrapperBase::Emplace{}, std::move(productOnHost));
        };
      } else {
        return nullptr;
      }
    }

    std::function<std::any(Queue&, edm::WrapperBase const&)> preTransformHtoD() const override {
      if constexpr (HasCopyToHost<T, Queue>) {
        // HasCopyToHost is required to get HostProductType, which is needed to
        // evaluate HasCopyToDevice<HostProductType, Queue>.
        using HostProductType = HostTypeOf<T, Queue>;
        if constexpr (HasCopyToDevice<HostProductType, Queue>) {
          using CopyT = cms::alpakatools::CopyToDevice<HostProductType>;
          using DeviceProductType = std::remove_cvref_t<decltype(CopyT::copyAsync(
              std::declval<Queue&>(), std::declval<HostProductType const&>()))>;
          static_assert(std::is_same_v<DeviceProductType, T>,
                        "CopyToDevice<HostTypeOf<T, Queue>>::copyAsync() must return a device product of type T!");

          return [](Queue& queue, edm::WrapperBase const& wb) -> std::any {
            auto const& hostProduct = dynamic_cast<edm::Wrapper<HostProductType> const&>(wb).bareProduct();
            auto productOnDevice = CopyT::copyAsync(queue, hostProduct);
            return std::make_shared<DeviceProductType>(std::move(productOnDevice));
          };
        }
      }
      return nullptr;
    }

    std::function<std::unique_ptr<edm::WrapperBase>(std::any const&, std::shared_ptr<EDMetadata>)> transformHtoD()
        const override {
      if constexpr (HasCopyToHost<T, Queue>) {
        using HostProductType = HostTypeOf<T, Queue>;
        if constexpr (HasCopyToDevice<HostProductType, Queue>) {
          using CopyT = cms::alpakatools::CopyToDevice<HostProductType>;
          using DeviceProductType = std::remove_cvref_t<decltype(CopyT::copyAsync(
              std::declval<Queue&>(), std::declval<HostProductType const&>()))>;
          static_assert(std::is_same_v<DeviceProductType, T>,
                        "CopyToDevice<HostTypeOf<T, Queue>>::copyAsync() must return a device product of type T!");

          return [](std::any const& cache, std::shared_ptr<EDMetadata> metadata) -> std::unique_ptr<edm::WrapperBase> {
            auto& productOnDevice = *std::any_cast<std::shared_ptr<DeviceProductType>>(cache);
            if constexpr (detail::useProductDirectly) {
              return std::make_unique<WrapperType>(edm::WrapperBase::Emplace{}, std::move(productOnDevice));
            } else {
              return std::make_unique<WrapperType>(
                  edm::WrapperBase::Emplace{}, std::move(metadata), std::move(productOnDevice));
            }
          };
        }
      }
      return nullptr;
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

#endif  // HeterogeneousCore_TrivialSerialisation_interface_alpaka_Serialiser_h
