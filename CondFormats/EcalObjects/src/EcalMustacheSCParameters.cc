#include "CondFormats/EcalObjects/interface/EcalMustacheSCParameters.h"

float EcalMustacheSCParameters::sqrtLogClustETuning() const { return sqrtLogClustETuning_; }

const EcalMustacheSCParameters::ParabolaParameters* EcalMustacheSCParameters::parabolaParameters(
    float log10ClustE, float absSeedEta) const {
  // assume the collection is lexicographically sorted in ascending ParabolaParameters.log10EMin and ascending ParabolaParameters.etaMin
  // find the matching log10EMin value
  auto it1 = std::lower_bound(parabolaParametersCollection_.begin(),
                              parabolaParametersCollection_.end(),
                              log10ClustE,
                              [](const EcalMustacheSCParameters::ParabolaParameters& params, const double var) {
                                return params.log10EMin < var;
                              });
  if (it1 != parabolaParametersCollection_.begin()) {
    --it1;
  }

  // find the matching log10EMin and etaMin entry going only up to the sets matching for log10ClustE
  const auto vars = std::make_pair(it1->log10EMin, absSeedEta);
  auto it2 = std::lower_bound(
      parabolaParametersCollection_.begin(),
      it1 + 1,
      vars,
      [](const EcalMustacheSCParameters::ParabolaParameters& params, const std::pair<double, double> vars) {
        return params.log10EMin < vars.first || params.etaMin < vars.second;
      });

  return (it2 != parabolaParametersCollection_.begin()) ? &*(it2 - 1) : nullptr;
}

void EcalMustacheSCParameters::print(std::ostream& out) const {
  out << "Mustache SC parameters:" << std::endl;
  out << " sqrtLogClustETuning: " << sqrtLogClustETuning_ << std::endl;
  out << " Parabola parameters are binned in " << parabolaParametersCollection_.size() << " (log10(E), |eta|) regions."
      << std::endl;
  for (const auto& params : parabolaParametersCollection_) {
    out << " Parameters for log10(E_min)=" << params.log10EMin << " and |eta_min|=" << params.etaMin << ":"
        << std::endl;

    out << "  pUp:" << std::endl;
    for (size_t i = 0; i < params.pUp.size(); ++i) {
      out << "   [" << i << "]: " << params.pUp[i] << std::endl;
    }

    out << "  pLow:" << std::endl;
    for (size_t i = 0; i < params.pLow.size(); ++i) {
      out << "   [" << i << "]: " << params.pLow[i] << std::endl;
    }

    out << "  w0Up:" << std::endl;
    for (size_t i = 0; i < params.w0Up.size(); ++i) {
      out << "   [" << i << "]: " << params.w0Up[i] << std::endl;
    }

    out << "  w1Up:" << std::endl;
    for (size_t i = 0; i < params.w1Up.size(); ++i) {
      out << "   [" << i << "]: " << params.w1Up[i] << std::endl;
    }

    out << "  w0Low:" << std::endl;
    for (size_t i = 0; i < params.w0Low.size(); ++i) {
      out << "   [" << i << "]: " << params.w0Low[i] << std::endl;
    }

    out << "  w1Low:" << std::endl;
    for (size_t i = 0; i < params.w1Low.size(); ++i) {
      out << "   [" << i << "]: " << params.w1Low[i] << std::endl;
    }
  }
}
