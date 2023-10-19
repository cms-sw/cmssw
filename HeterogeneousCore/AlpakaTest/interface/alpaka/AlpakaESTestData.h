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
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
