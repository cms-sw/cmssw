#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotObjects.h"

#include <cmath>
#include <iostream>

// Get SigmaX and SigmaY:
// - directly fSigmaX if >= 0 (in case of Gaussian Smearing)
// - else from LPC-like calculation (in case of BetaFunc Smearing)
double SimBeamSpotObjects::sigmaX() const {
  if (fSigmaX >= 0.)  // Gaussian smearing
    return fSigmaX;
  else  // BetaFunc smearing
    return (1 / std::sqrt(2)) * std::sqrt(femittance * fbetastar);
}

double SimBeamSpotObjects::sigmaY() const {
  if (fSigmaY >= 0.)  // Gaussian smearing
    return fSigmaY;
  else  // BetaFunc smearing
    return (1 / std::sqrt(2)) * std::sqrt(femittance * fbetastar);
}

// Printout SimBeamSpotObjects
void SimBeamSpotObjects::print(std::stringstream& ss) const {
  ss << "-----------------------------------------------------\n"
     << "              Sim Beam Spot Data\n\n"
     << "       X0     = " << x() << " [cm]\n"
     << "       Y0     = " << y() << " [cm]\n"
     << "       Z0     = " << z() << " [cm]\n"
     << "    MeanX     = " << meanX() << " [cm]\n"
     << "    MeanY     = " << meanY() << " [cm]\n"
     << "    MeanZ     = " << meanZ() << " [cm]\n"
     << " Sigma X0     = " << sigmaX() << " [cm]\n"
     << " Sigma Y0     = " << sigmaY() << " [cm]\n"
     << " Sigma Z0     = " << sigmaZ() << " [cm]\n"
     << " Beta star    = " << betaStar() << " [cm]\n"
     << " Emittance X  = " << emittance() << " [cm]\n"
     << " Phi          = " << phi() << " [radians]\n"
     << " Alpha        = " << alpha() << " [radians]\n"
     << " TimeOffset   = " << timeOffset() << " [ns]\n"
     << "-----------------------------------------------------\n\n";
}

std::ostream& operator<<(std::ostream& os, SimBeamSpotObjects beam) {
  std::stringstream ss;
  beam.print(ss);
  os << ss.str();
  return os;
}
