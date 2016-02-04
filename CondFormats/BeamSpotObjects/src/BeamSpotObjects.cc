#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

#include <iostream>

void BeamSpotObjects::print(std::stringstream& ss) const {
	  ss << "-----------------------------------------------------\n"
	     << "              Beam Spot Data\n\n"
		 << " Beam type    = " << GetBeamType() << "\n"
	     << "       X0     = " << GetX() << " +/- " << GetXError() << " [cm]\n"
	     << "       Y0     = " << GetY() << " +/- " << GetYError() << " [cm]\n"
	     << "       Z0     = " << GetZ() << " +/- " << GetZError() << " [cm]\n"
	     << " Sigma Z0     = " << GetSigmaZ() << " +/- " << GetSigmaZError() << " [cm]\n"
	     << " dxdz         = " << Getdxdz() << " +/- " << GetdxdzError() << " [radians]\n"
	     << " dydz         = " << Getdydz() << " +/- " << GetdydzError() << " [radians]\n"
	     << " Beam Width X = " << GetBeamWidthX() << " +/- " << GetBeamWidthXError() << " [cm]\n"
		 << " Beam Width Y = " << GetBeamWidthY() << " +/- " << GetBeamWidthYError() << " [cm]\n"
		 << " Emittance X  = " << GetEmittanceX() << " [cm]\n"
		 << " Emittance Y  = " << GetEmittanceY() << " [cm]\n"
		 << " Beta star    = " << GetBetaStar() << " [cm]\n"
	     << "-----------------------------------------------------\n\n";
}

std::ostream& operator<< ( std::ostream& os, BeamSpotObjects beam ) {
  std::stringstream ss;
  beam.print(ss);
  os << ss.str();
  return os;
}
