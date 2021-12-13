#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

#include <iostream>

void BeamSpotObjects::print(std::stringstream& ss) const {
  ss << "-----------------------------------------------------\n"
     << "              Beam Spot Data\n\n"
     << " Beam type    = " << beamType() << "\n"
     << "       X0     = " << x() << " +/- " << xError() << " [cm]\n"
     << "       Y0     = " << y() << " +/- " << yError() << " [cm]\n"
     << "       Z0     = " << z() << " +/- " << zError() << " [cm]\n"
     << " Sigma Z0     = " << sigmaZ() << " +/- " << sigmaZError() << " [cm]\n"
     << " dxdz         = " << dxdz() << " +/- " << dxdzError() << " [radians]\n"
     << " dydz         = " << dydz() << " +/- " << dydzError() << " [radians]\n"
     << " Beam Width X = " << beamWidthX() << " +/- " << beamWidthXError() << " [cm]\n"
     << " Beam Width Y = " << beamWidthY() << " +/- " << beamWidthYError() << " [cm]\n"
     << " Emittance X  = " << emittanceX() << " [cm]\n"
     << " Emittance Y  = " << emittanceY() << " [cm]\n"
     << " Beta star    = " << betaStar() << " [cm]\n"
     << "-----------------------------------------------------\n\n";
}

std::ostream& operator<<(std::ostream& os, BeamSpotObjects beam) {
  std::stringstream ss;
  beam.print(ss);
  os << ss.str();
  return os;
}
