#ifndef DataFormats_Portable_interface_alpaka_PortableCollection_h
#define DataFormats_Portable_interface_alpaka_PortableCollection_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"

// This header is not used by PortableCollection, but is included here to automatically
// provide its content to users of ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection.
#include "HeterogeneousCore/AlpakaInterface/interface/AssertDeviceMatchesHostCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  // ... or any other CPU-based accelerators

  // generic SoA-based product in host memory
  template <typename T>
  using PortableCollection = ::PortableHostCollection<T>;

#else

  // generic SoA-based product in device memory
  template <typename T>
  using PortableCollection = ::PortableDeviceCollection<T, Device>;

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

namespace traits {

  // specialise the trait for the device provided by the ALPAKA_ACCELERATOR_NAMESPACE
  template <typename T>
  class PortableCollectionTrait<T, ALPAKA_ACCELERATOR_NAMESPACE::Device> {
    using CollectionType = ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<T>;
  };

}  // namespace traits

namespace cms::alpakatools {
  template <typename TLayout, typename TDevice>
  struct CopyToHost<PortableDeviceCollection<TLayout, TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PortableDeviceCollection<TLayout, TDevice> const& srcData) {
      PortableHostCollection<TLayout> dstData(srcData->metadata().size(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };

  template <typename TLayout>
  struct CopyToDevice<PortableHostCollection<TLayout>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, PortableHostCollection<TLayout> const& srcData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;
      PortableDeviceCollection<TLayout, TDevice> dstData(srcData->metadata().size(), queue);
      alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
      return dstData;
    }
  };
}  // namespace cms::alpakatools

#endif  // DataFormats_Portable_interface_alpaka_PortableCollection_h
