#include "DataFormats/RPCDigi/interface/IRPCDigiTime.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"

IRPCDigiTime::IRPCDigiTime(const IRPCDigi& adigi) : theDigi(adigi) {}

float IRPCDigiTime::time() { return (timeLR() + timeHR()) / 2.; }

float IRPCDigiTime::coordinateY() {
  const double signal_speed = 0.66 * CLHEP::c_light * CLHEP::ns / CLHEP::cm;
  ;                                                  // CLHEP::c_light is in [mm/ns] we need [cm/ns] here
  return signal_speed * (timeLR() - timeHR()) / 2.;  // the difference between two measured times in [cm]
}

float IRPCDigiTime::timeLR() { return TDC2Time(theDigi.bxLR(), theDigi.sbxLR(), theDigi.fineLR()); }

float IRPCDigiTime::timeHR() { return TDC2Time(theDigi.bxHR(), theDigi.sbxHR(), theDigi.fineHR()); }

float IRPCDigiTime::TDC2Time(int BX, int SBX, int FT) { return 25. * BX + 2.5 * SBX + 0.2 * FT; }
