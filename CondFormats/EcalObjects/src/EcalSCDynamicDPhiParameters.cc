#include "CondFormats/EcalObjects/interface/EcalSCDynamicDPhiParameters.h"

const EcalSCDynamicDPhiParameters::DynamicDPhiParameters* EcalSCDynamicDPhiParameters::dynamicDPhiParameters(
    double clustE, double absSeedEta) const {
  // assume the collection is lexicographically sorted in ascending DynamicDPhiParams.eMin and ascending DynamicDPhiParams.etaMin
  // find the matching eMin value
  auto it1 = std::lower_bound(dynamicDPhiParametersCollection_.begin(),
                              dynamicDPhiParametersCollection_.end(),
                              clustE,
                              [](const EcalSCDynamicDPhiParameters::DynamicDPhiParameters& params, const double var) {
                                return params.eMin < var;
                              });
  if (it1 != dynamicDPhiParametersCollection_.begin()) {
    --it1;
  }

  // find the matching eMin and etaMin entry going only up to the sets matching for clustE
  const auto vars = std::make_pair(it1->eMin, absSeedEta);
  auto it2 = std::lower_bound(
      dynamicDPhiParametersCollection_.begin(),
      it1 + 1,
      vars,
      [](const EcalSCDynamicDPhiParameters::DynamicDPhiParameters& params, const std::pair<double, double> vars) {
        return params.eMin < vars.first || params.etaMin < vars.second;
      });

  return (it2 != dynamicDPhiParametersCollection_.begin()) ? &*(it2 - 1) : nullptr;
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
