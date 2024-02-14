#ifndef HeterogeneousCore_AlpakaTest_interface_AlpakaESTestSoA_h
#define HeterogeneousCore_AlpakaTest_interface_AlpakaESTestSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace cms::alpakatest {
  // PortableCollection-based model
  GENERATE_SOA_LAYOUT(AlpakaESTestSoALayoutA, SOA_COLUMN(int, z))
  GENERATE_SOA_LAYOUT(AlpakaESTestSoALayoutC, SOA_COLUMN(int, x))
  GENERATE_SOA_LAYOUT(AlpakaESTestSoALayoutD, SOA_COLUMN(int, y))
  GENERATE_SOA_LAYOUT(AlpakaESTestSoALayoutE, SOA_COLUMN(float, val), SOA_COLUMN(int, ind))
  GENERATE_SOA_LAYOUT(AlpakaESTestSoALayoutEData, SOA_COLUMN(float, val2))

  using AlpakaESTestSoAA = AlpakaESTestSoALayoutA<>;
  using AlpakaESTestSoAC = AlpakaESTestSoALayoutC<>;
  using AlpakaESTestSoAD = AlpakaESTestSoALayoutD<>;
  using AlpakaESTestSoAE = AlpakaESTestSoALayoutE<>;
  using AlpakaESTestSoAEData = AlpakaESTestSoALayoutEData<>;
}  // namespace cms::alpakatest

#endif
