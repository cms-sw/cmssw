#ifndef DataFormats_Portable_interface_PortableCollection_h
#define DataFormats_Portable_interface_PortableCollection_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/concepts.h"

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

}  // namespace traits

// type alias for a generic SoA-based product
template <typename T, typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
using PortableCollection = typename traits::PortableCollectionTrait<T, TDev>::CollectionType;

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

}  // namespace cms::alpakatools

#endif  // DataFormats_Portable_interface_PortableCollection_h
