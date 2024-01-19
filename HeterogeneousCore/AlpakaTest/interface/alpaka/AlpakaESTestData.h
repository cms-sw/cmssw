#ifndef HeterogeneousCore_AlpakaTest_interface_alpaka_AlpakaESTestData_h
#define HeterogeneousCore_AlpakaTest_interface_alpaka_AlpakaESTestData_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  // PortableCollection-based model
  using AlpakaESTestDataAHost = cms::alpakatest::AlpakaESTestDataAHost;
  using AlpakaESTestDataADevice = PortableCollection<cms::alpakatest::AlpakaESTestSoAA>;

  using AlpakaESTestDataCHost = cms::alpakatest::AlpakaESTestDataCHost;
  using AlpakaESTestDataCDevice = PortableCollection<cms::alpakatest::AlpakaESTestSoAC>;

  using AlpakaESTestDataDHost = cms::alpakatest::AlpakaESTestDataDHost;
  using AlpakaESTestDataDDevice = PortableCollection<cms::alpakatest::AlpakaESTestSoAD>;

  using AlpakaESTestDataEHost = cms::alpakatest::AlpakaESTestDataEHost;
  using AlpakaESTestDataEDevice = cms::alpakatest::AlpakaESTestDataE<Device>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

// check that the portable device collections for the host device are the same as the portable host collections
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(AlpakaESTestDataADevice, cms::alpakatest::AlpakaESTestDataAHost);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(AlpakaESTestDataCDevice, cms::alpakatest::AlpakaESTestDataCHost);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(AlpakaESTestDataDDevice, cms::alpakatest::AlpakaESTestDataDHost);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(AlpakaESTestDataEDevice, cms::alpakatest::AlpakaESTestDataEHost);

#endif  // HeterogeneousCore_AlpakaTest_interface_alpaka_AlpakaESTestData_h
