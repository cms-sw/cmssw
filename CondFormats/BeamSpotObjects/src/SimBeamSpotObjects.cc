#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotObjects.h"

#include <iostream>

void SimBeamSpotObjects::print(std::stringstream& ss) const {
  ss << "-----------------------------------------------------\n"
     << "              Sim Beam Spot Data\n\n"
     << "       X0     = " << x() << " [cm]\n"
     << "       Y0     = " << y() << " [cm]\n"
     << "       Z0     = " << z() << " [cm]\n"
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
