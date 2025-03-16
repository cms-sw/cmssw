#ifndef DataFormats_HeterogeneousTutorial_interface_JetsSoA_h
#define DataFormats_HeterogeneousTutorial_interface_JetsSoA_h

#include <ostream>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace tutorial {

  GENERATE_SOA_LAYOUT(JetsSoALayout,
                      // columns: one value per element
                      SOA_COLUMN(float, pt),
                      SOA_COLUMN(float, eta),
                      SOA_COLUMN(float, phi),
                      SOA_COLUMN(float, mass))

  using JetsSoA = JetsSoALayout<>;

  std::ostream& operator<<(std::ostream& out, JetsSoA::View::const_element const& jet);

}  // namespace tutorial

#endif  // DataFormats_HeterogeneousTutorial_interface_JetsSoA_h
