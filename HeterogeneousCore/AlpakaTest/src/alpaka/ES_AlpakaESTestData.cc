#include "HeterogeneousCore/AlpakaCore/interface/alpaka/typelookup.h"

// Model 1
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"
TYPELOOKUP_ALPAKA_DATA_REG(AlpakaESTestDataADevice);
TYPELOOKUP_ALPAKA_DATA_REG(AlpakaESTestDataCDevice);
TYPELOOKUP_ALPAKA_DATA_REG(AlpakaESTestDataDDevice);

// Model 2
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestData.h"
TYPELOOKUP_ALPAKA_TEMPLATED_DATA_REG(cms::alpakatest::AlpakaESTestDataB);
