#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"

#include <iostream>

void BeamSpotOnlineObjects::print(std::stringstream& ss) const {
  ss << "-----------------------------------------------------\n"
     << "              BeamSpotOnline Data\n\n"
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
     << " Last Lumi    = " << GetLastAnalyzedLumi() << "\n"
     << " Last Run     = " << GetLastAnalyzedRun() << "\n"
     << " Last Fill    = " << GetLastAnalyzedFill() << "\n"
     << " Target lumi  = " << GetTargetLumi() << "\n"
     << "-----------------------------------------------------\n\n";
}
