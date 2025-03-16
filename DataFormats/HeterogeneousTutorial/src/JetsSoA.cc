#include <ostream>

#include <fmt/format.h>

#include "DataFormats/HeterogeneousTutorial/interface/JetsSoA.h"

namespace tutorial {

  std::ostream& operator<<(std::ostream& out, JetsSoA::View::const_element const& jet) {
    out << fmt::format("SoA jet (pt, eta, phi) {:8.3f}, {:6.3f}, {:6.3f}", jet.pt(), jet.eta(), jet.phi());
    return out;
  }

}  // namespace tutorial
