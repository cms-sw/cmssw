#ifndef DataFormats_Portable_interface_alpaka_PortableProduct_h
#define DataFormats_Portable_interface_alpaka_PortableProduct_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableProduct.h"
#include "DataFormats/Portable/interface/PortableHostProduct.h"
#include "DataFormats/Portable/interface/PortableDeviceProduct.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  // ... or any other CPU-based accelerators

  // generic SoA-based product in host memory
  template <typename T>
  using PortableProduct = ::PortableHostProduct<T>;

#else

  // generic SoA-based product in device memory
  template <typename T>
  using PortableProduct = ::PortableDeviceProduct<T, Device>;

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace traits {

  // specialise the trait for the device provided by the ALPAKA_ACCELERATOR_NAMESPACE
  template <typename T>
  class PortableProductTrait<T, ALPAKA_ACCELERATOR_NAMESPACE::Device> {
    using ProductType = ALPAKA_ACCELERATOR_NAMESPACE::PortableProduct<T>;
  };

}  // namespace traits

namespace cms::alpakatools {
  template <typename TProduct, typename TDevice>
  struct CopyToHost<PortableDeviceProduct<TProduct, TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PortableDeviceProduct<TProduct, TDevice> const& srcData) {
      PortableHostProduct<TProduct> dstData(queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };

  template <typename TProduct>
  struct CopyToDevice<PortableHostProduct<TProduct>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PortableHostProduct<TProduct> const& srcData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;
      PortableDeviceProduct<TProduct, TDevice> dstData(queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

#endif  // DataFormats_Portable_interface_alpaka_PortableProduct_h
