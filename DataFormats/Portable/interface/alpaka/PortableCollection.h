#ifndef DataFormats_Portable_interface_alpaka_PortableDeviceCollection_h
#define DataFormats_Portable_interface_alpaka_PortableDeviceCollection_h

#include <optional>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

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

// overloads must be defined in the global namespace because Portable{Device,Host}Collection are
template <typename TQueue, typename TLayout>
auto copyToHostAsync(TQueue& queue,
                     PortableDeviceCollection<TLayout, typename alpaka::trait::DevType<TQueue>::type> const& srcData) {
  PortableHostCollection<TLayout> dstData(srcData->metadata().size(), queue);
  alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
  return dstData;
}

template <typename TQueue, typename TLayout>
auto copyToDeviceAsync(TQueue& queue, PortableHostCollection<TLayout> const& srcData) {
  PortableDeviceCollection<TLayout, typename alpaka::trait::DevType<TQueue>::type> dstData(srcData->metadata().size(),
                                                                                           queue);
  alpaka::memcpy(queue, dstData.buffer(), srcData.buffer());
  return dstData;
}

#endif  // DataFormats_Portable_interface_alpaka_PortableDeviceCollection_h
