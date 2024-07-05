#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotHLLHCObjects.h"

#include <iostream>

void SimBeamSpotHLLHCObjects::print(std::stringstream& ss) const {
  ss << "------------------------------------------------------------------------\n"
     << "              Sim Beam Spot HL LHC Data\n\n"
     << "    MeanX                  = " << meanX() << " [cm]\n"
     << "    MeanY                  = " << meanY() << " [cm]\n"
     << "    MeanZ                  = " << meanZ() << " [cm]\n"
     << " E Proton                  = " << eProton() << " [GeV]\n"
     << " Crab Frequency            = " << crabFrequency() << " [MHz]\n"
     << " 800 MHz RF                ? " << rf800() << "\n"
     << " Crossing Angle            = " << crossingAngle() << " [urad]\n"
     << " Crabbing Angle Crossing   = " << crabbingAngleCrossing() << " [urad]\n"
     << " Crabbing Angle Separation = " << crabbingAngleSeparation() << " [urad]\n"
     << " Beta Crossing Plane       = " << betaCrossingPlane() << " [m]\n"
     << " Beta Separation Plane     = " << betaSeparationPlane() << " [m]\n"
     << " Horizontal Emittance      = " << horizontalEmittance() << " [mm]\n"
     << " Vertical Emittance        = " << verticalEmittance() << " [mm]\n"
     << " Bunch Lenght              = " << bunchLenght() << " [m]\n"
     << " TimeOffset                = " << timeOffset() << " [ns]\n"
     << "------------------------------------------------------------------------\n\n";
}

std::ostream& operator<<(std::ostream& os, SimBeamSpotHLLHCObjects beam) {
  std::stringstream ss;
  beam.print(ss);
  os << ss.str();
  return os;
}
