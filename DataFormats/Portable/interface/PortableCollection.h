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
  template <typename TDev, typename T, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
  struct PortableCollectionTrait {
    using CollectionType = PortableDeviceCollection<TDev, T>;
  };

  // specialise for host device
  template <typename T>
  struct PortableCollectionTrait<alpaka_common::DevHost, T> {
    using CollectionType = PortableHostCollection<T>;
  };

}  // namespace traits

// type alias for a generic SoA-based product
template <typename TDev, typename T, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
using PortableCollection = typename traits::PortableCollectionTrait<TDev, T>::CollectionType;

// define how to copy PortableCollection between host and device
namespace cms::alpakatools {
  template <typename TDevice, typename TLayout>
    requires alpaka::isDevice<TDevice>
  struct CopyToHost<PortableDeviceCollection<TDevice, TLayout>> {
    template <typename TQueue>
      requires alpaka::isQueue<TQueue>
    static auto copyAsync(TQueue& queue, PortableDeviceCollection<TDevice, TLayout> const& srcData) {
      PortableHostCollection<TLayout> dstData(queue, srcData->metadata().size());
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };

  template <typename TLayout>
  struct CopyToDevice<PortableHostCollection<TLayout>> {
    template <cms::alpakatools::NonCPUQueue TQueue>
    static auto copyAsync(TQueue& queue, PortableHostCollection<TLayout> const& srcData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;
      PortableDeviceCollection<TDevice, TLayout> dstData(queue, srcData->metadata().size());
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };

}  // namespace cms::alpakatools

#endif  // DataFormats_Portable_interface_PortableCollection_h
