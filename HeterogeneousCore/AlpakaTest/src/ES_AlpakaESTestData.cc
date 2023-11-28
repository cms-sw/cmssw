#include "HeterogeneousCore/AlpakaTest/interface/AlpakaESTestData.h"
#include "FWCore/Utilities/interface/typelookup.h"

// PortableCollection-based model
TYPELOOKUP_DATA_REG(cms::alpakatest::AlpakaESTestDataAHost);
TYPELOOKUP_DATA_REG(cms::alpakatest::AlpakaESTestDataCHost);
TYPELOOKUP_DATA_REG(cms::alpakatest::AlpakaESTestDataDHost);

// Template-over-device model
TYPELOOKUP_DATA_REG(cms::alpakatest::AlpakaESTestDataB<alpaka_common::DevHost>);
