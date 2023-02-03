#ifndef HeterogeneousCore_AlpakaTest_plugins_alpaka_testESAlgoAsync_h
#define HeterogeneousCore_AlpakaTest_plugins_alpaka_testESAlgoAsync_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestData.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  AlpakaESTestDataDDevice testESAlgoAsync(Queue& queue,
                                          AlpakaESTestDataADevice const& dataA,
                                          cms::alpakatest::AlpakaESTestDataB<Device> const& dataB);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // HeterogeneousCore_AlpakaTest_plugins_alpaka_TestAlgo_h
