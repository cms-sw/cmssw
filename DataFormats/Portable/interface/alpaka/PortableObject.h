#ifndef DataFormats_Portable_interface_alpaka_PortableObject_h
#define DataFormats_Portable_interface_alpaka_PortableObject_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableObject.h"
#include "DataFormats/Portable/interface/PortableHostObject.h"
#include "DataFormats/Portable/interface/PortableDeviceObject.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

// This header is not used by PortableObject, but is included here to automatically
// provide its content to users of ALPAKA_ACCELERATOR_NAMESPACE::PortableObject.
#include "HeterogeneousCore/AlpakaInterface/interface/AssertDeviceMatchesHostCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  // ... or any other CPU-based accelerators

  // generic SoA-based product in host memory
  template <typename T>
  using PortableObject = ::PortableHostObject<T>;

#else

  // generic SoA-based product in device memory
  template <typename T>
  using PortableObject = ::PortableDeviceObject<T, Device>;

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace traits {

  // specialise the trait for the device provided by the ALPAKA_ACCELERATOR_NAMESPACE
  template <typename T>
  class PortableObjectTrait<T, ALPAKA_ACCELERATOR_NAMESPACE::Device> {
    using ProductType = ALPAKA_ACCELERATOR_NAMESPACE::PortableObject<T>;
  };

}  // namespace traits

namespace cms::alpakatools {
  template <typename TProduct, typename TDevice>
  struct CopyToHost<PortableDeviceObject<TProduct, TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PortableDeviceObject<TProduct, TDevice> const& srcData) {
      PortableHostObject<TProduct> dstData(queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };

  template <typename TProduct>
  struct CopyToDevice<PortableHostObject<TProduct>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PortableHostObject<TProduct> const& srcData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;
      PortableDeviceObject<TProduct, TDevice> dstData(queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

#endif  // DataFormats_Portable_interface_alpaka_PortableObject_h
