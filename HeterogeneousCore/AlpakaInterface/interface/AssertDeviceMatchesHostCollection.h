#ifndef HeterogeneousCore_AlpakaInterface_interface_AssertDeviceMatchesHostCollection_h
#define HeterogeneousCore_AlpakaInterface_interface_AssertDeviceMatchesHostCollection_h

#include <type_traits>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

// check that the portable device collection for the host device is the same as the portable host collection
#define ASSERT_DEVICE_MATCHES_HOST_COLLECTION(DEVICE_COLLECTION, HOST_COLLECTION)       \
  static_assert(std::is_same_v<alpaka_serial_sync::DEVICE_COLLECTION, HOST_COLLECTION>, \
                "The device collection for the host device and the host collection must be the same type!");

#else

// the portable device collections for the non-host devices do not require any checks
#define ASSERT_DEVICE_MATCHES_HOST_COLLECTION(DEVICE_COLLECTION, HOST_COLLECTION)

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#endif  // HeterogeneousCore_AlpakaInterface_interface_AssertDeviceMatchesHostCollection_h
