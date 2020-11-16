#include "CondFormats/EcalObjects/interface/EcalSCDynamicDPhiParameters.h"

EcalSCDynamicDPhiParameters::DynamicDPhiParameters EcalSCDynamicDPhiParameters::dynamicDPhiParameters(
    double clustE, double absSeedEta) const {
  // assume the collection is sorted in descending DynamicDPhiParams.etaMin and descending DynamicDPhiParams.eMin
  for (const auto &dynamicDPhiParams : dynamicDPhiParametersCollection_) {
    if (clustE < dynamicDPhiParams.eMin || absSeedEta < dynamicDPhiParams.etaMin) {
      continue;
    } else {
      return dynamicDPhiParams;
    }
  }
  return EcalSCDynamicDPhiParameters::DynamicDPhiParameters();
}

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
