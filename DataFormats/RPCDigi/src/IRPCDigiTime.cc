#include "DataFormats/RPCDigi/interface/IRPCDigiTime.h"

IRPCDigiTime::IRPCDigiTime(const IRPCDigi& adigi) : theDigi(adigi) {}

float IRPCDigiTime::time() { return (timeLR() + timeHR()) / 2.; }

float IRPCDigiTime::coordinateY() {
  const double signal_speed = 0.66 * 299792458e-7;  //signal propagation speed [cm/ns]
  return signal_speed * (timeLR() - timeHR()) / 2.;
}

float IRPCDigiTime::timeLR() { return TDC2Time(theDigi.bxLR(), theDigi.sbxLR(), theDigi.fineLR()); }

float IRPCDigiTime::timeHR() { return TDC2Time(theDigi.bxHR(), theDigi.sbxHR(), theDigi.fineHR()); }

float IRPCDigiTime::TDC2Time(int BX, int SBX, int FT) { return 25. * BX + 2.5 * SBX + 0.2 * FT; }
