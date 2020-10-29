#include "CondFormats/EcalObjects/interface/EcalSCDynamicDPhiParameters.h"

void EcalSCDynamicDPhiParameters::print(std::ostream& out) const {
  out << "SC dynamic dPhi parameters:" << std::endl;
  out << " Parameters are binned in " << dynamicDPhiParametersCollection_.size() << " (E, |eta|) regions." << std::endl;
  for (const auto& params : dynamicDPhiParametersCollection_) {
    out << " Parameters for E_min=" << params.eMin << " and |eta_min|=" << params.etaMin << ":" << std::endl;
    out << "  yoffset:    " << params.yoffset << std::endl;
    out << "  scale:      " << params.scale << std::endl;
    out << "  xoffset:    " << params.xoffset << std::endl;
    out << "  width:      " << params.width << std::endl;
    out << "  saturation: " << params.saturation << std::endl;
    out << "  cutoff:     " << params.cutoff << std::endl;
  }
}
