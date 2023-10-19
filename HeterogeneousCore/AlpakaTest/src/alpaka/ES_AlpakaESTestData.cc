#include "HeterogeneousCore/AlpakaCore/interface/alpaka/typelookup.h"

// PortableCollection-based model
#include "HeterogeneousCore/AlpakaTest/interface/alpaka/AlpakaESTestData.h"
TYPELOOKUP_ALPAKA_DATA_REG(AlpakaESTestDataADevice);
TYPELOOKUP_ALPAKA_DATA_REG(AlpakaESTestDataCDevice);
TYPELOOKUP_ALPAKA_DATA_REG(AlpakaESTestDataDDevice);

// Template-over-device model
#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestData.h"
TYPELOOKUP_ALPAKA_TEMPLATED_DATA_REG(cms::alpakatest::AlpakaESTestDataB);
