#ifndef DataFormats_Portable_interface_PortableObject_h
#define DataFormats_Portable_interface_PortableObject_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "DataFormats/Portable/interface/PortableDeviceObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/concepts.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace traits {

  // trait for a generic struct-based product
  template <typename T, typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
  struct PortableObjectTrait {
    using ProductType = PortableDeviceObject<T, TDev>;
  };

  // specialise for host device
  template <typename T>
  struct PortableObjectTrait<T, alpaka_common::DevHost> {
    using ProductType = PortableHostObject<T>;
  };

}  // namespace traits

// type alias for a generic struct-based product
template <typename T, typename TDev, typename = std::enable_if_t<alpaka::isDevice<TDev>>>
using PortableObject = typename traits::PortableObjectTrait<T, TDev>::ProductType;

// define how to copy PortableObject between host and device
namespace cms::alpakatools {
  template <typename TProduct, typename TDevice>
    requires alpaka::isDevice<TDevice>
  struct CopyToHost<PortableDeviceObject<TProduct, TDevice>> {
    template <typename TQueue>
      requires alpaka::isQueue<TQueue>
    static auto copyAsync(TQueue& queue, PortableDeviceObject<TProduct, TDevice> const& srcData) {
      PortableHostObject<TProduct> dstData(queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };

  template <typename TProduct>
  struct CopyToDevice<PortableHostObject<TProduct>> {
    template <cms::alpakatools::NonCPUQueue TQueue>
    static auto copyAsync(TQueue& queue, PortableHostObject<TProduct> const& srcData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;
      PortableDeviceObject<TProduct, TDevice> dstData(queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

#endif  // DataFormats_Portable_interface_PortableObject_h
