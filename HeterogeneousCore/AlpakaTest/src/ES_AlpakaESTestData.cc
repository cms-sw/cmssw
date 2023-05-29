#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestData.h"
#include "FWCore/Utilities/interface/typelookup.h"

// Model 1
TYPELOOKUP_DATA_REG(cms::alpakatest::AlpakaESTestDataAHost);
TYPELOOKUP_DATA_REG(cms::alpakatest::AlpakaESTestDataCHost);
TYPELOOKUP_DATA_REG(cms::alpakatest::AlpakaESTestDataDHost);

// Model 2
TYPELOOKUP_DATA_REG(cms::alpakatest::AlpakaESTestDataB<alpaka_common::DevHost>);
