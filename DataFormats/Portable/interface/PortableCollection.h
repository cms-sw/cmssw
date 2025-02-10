#ifndef DataFormats_Portable_interface_PortableCollection_h
#define DataFormats_Portable_interface_PortableCollection_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/concepts.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace traits {

  // trait for a generic SoA-based product
  template <typename T, typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
  struct PortableCollectionTrait {
    using CollectionType = PortableDeviceCollection<T, TDev>;
  };

  // specialise for host device
  template <typename T>
  struct PortableCollectionTrait<T, alpaka_common::DevHost> {
    using CollectionType = PortableHostCollection<T>;
  };

  // trait for a generic multi-SoA-based product
  template <typename TDev, typename T0, typename... Args>
  struct PortableMultiCollectionTrait {
    using CollectionType = PortableDeviceMultiCollection<TDev, T0, Args...>;
  };

  // specialise for host device
  template <typename T0, typename... Args>
  struct PortableMultiCollectionTrait<alpaka_common::DevHost, T0, Args...> {
    using CollectionType = PortableHostMultiCollection<T0, Args...>;
  };

}  // namespace traits

// type alias for a generic SoA-based product
template <typename T, typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
using PortableCollection = typename traits::PortableCollectionTrait<T, TDev>::CollectionType;

// type alias for a generic SoA-based product
template <typename TDev, typename T0, typename... Args>
using PortableMultiCollection = typename traits::PortableMultiCollectionTrait<TDev, T0, Args...>::CollectionType;

// define how to copy PortableCollection between host and device
namespace cms::alpakatools {
  template <typename TLayout, typename TDevice>
    requires alpaka::isDevice<TDevice>
  struct CopyToHost<PortableDeviceCollection<TLayout, TDevice>> {
    template <typename TQueue>
      requires alpaka::isQueue<TQueue>
    static auto copyAsync(TQueue& queue, PortableDeviceCollection<TLayout, TDevice> const& srcData) {
      PortableHostCollection<TLayout> dstData(srcData->metadata().size(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };

  template <typename TDev, typename T0, typename... Args>
    requires alpaka::isDevice<TDev>
  struct CopyToHost<PortableDeviceMultiCollection<TDev, T0, Args...>> {
    template <typename TQueue>
      requires alpaka::isQueue<TQueue>
    static auto copyAsync(TQueue& queue, PortableDeviceMultiCollection<TDev, T0, Args...> const& srcData) {
      PortableHostMultiCollection<T0, Args...> dstData(srcData.sizes(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };

  template <typename TLayout>
  struct CopyToDevice<PortableHostCollection<TLayout>> {
    template <cms::alpakatools::NonCPUQueue TQueue>
    static auto copyAsync(TQueue& queue, PortableHostCollection<TLayout> const& srcData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;
      PortableDeviceCollection<TLayout, TDevice> dstData(srcData->metadata().size(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };

  template <typename T0, typename... Args>
  struct CopyToDevice<PortableHostMultiCollection<T0, Args...>> {
    template <cms::alpakatools::NonCPUQueue TQueue>
    static auto copyAsync(TQueue& queue, PortableHostMultiCollection<T0, Args...> const& srcData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;
      PortableDeviceMultiCollection<TDevice, T0, Args...> dstData(srcData.sizes(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

#endif  // DataFormats_Portable_interface_PortableCollection_h
