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

  using AlpakaESTestSoAA = AlpakaESTestSoALayoutA<>;
  using AlpakaESTestSoAC = AlpakaESTestSoALayoutC<>;
  using AlpakaESTestSoAD = AlpakaESTestSoALayoutD<>;
}  // namespace cms::alpakatest

#endif
