#ifndef HeterogeneousCore_Tutorial_interface_JetsSelectionSoA_h
#define HeterogeneousCore_Tutorial_interface_JetsSelectionSoA_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace tutorial {

  GENERATE_SOA_LAYOUT(JetsSelectionSoALayout,
                      // columns: one value per element
                      SOA_COLUMN(bool, valid))

  using JetsSelectionSoA = JetsSelectionSoALayout<>;

}  // namespace tutorial

#endif  // HeterogeneousCore_Tutorial_interface_JetsSelectionSoA_h
