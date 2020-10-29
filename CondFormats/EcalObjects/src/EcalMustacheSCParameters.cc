#include "CondFormats/EcalObjects/interface/EcalMustacheSCParameters.h"

void EcalMustacheSCParameters::print(std::ostream& out) const {
  out << "Mustache SC parameters:" << std::endl;
  out << " sqrtLogClustETuning: " << std::endl;
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
