#ifndef DataFormats_Portable_interface_alpaka_PortableDeviceCollection_h
#define DataFormats_Portable_interface_alpaka_PortableDeviceCollection_h

#include <optional>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/TransferToHost.h"

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
  // TODO: Is this the right place for the specialization? Or should it be in PortableDeviceProduct?
  template <typename T>
  struct TransferToHost<ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<T>> {
    using HostDataType = ::PortableHostCollection<T>;

    template <typename TQueue>
    static HostDataType transferAsync(TQueue& queue,
                                      ALPAKA_ACCELERATOR_NAMESPACE::PortableCollection<T> const& deviceData) {
      HostDataType hostData(deviceData->metadata().size(), queue);
      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
      return hostData;
    }
  };
}  // namespace cms::alpakatools

#endif  // DataFormats_Portable_interface_alpaka_PortableDeviceCollection_h
