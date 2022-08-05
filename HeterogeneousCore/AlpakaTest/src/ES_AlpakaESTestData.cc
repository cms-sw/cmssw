#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestData.h"
#include "FWCore/Utilities/interface/typelookup.h"

// Model 2
TYPELOOKUP_DATA_REG(cms::alpakatest::AlpakaESTestDataB<alpaka_common::DevHost>);

// Model 3
TYPELOOKUP_DATA_REG(cms::alpakatest::AlpakaESTestDataCHost);
TYPELOOKUP_DATA_REG(cms::alpakatest::AlpakaESTestDataDHost);
