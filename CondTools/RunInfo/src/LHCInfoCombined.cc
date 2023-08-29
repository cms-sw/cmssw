#include "CondTools/RunInfo/interface/LHCInfoCombined.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

LHCInfoCombined::LHCInfoCombined(const LHCInfo& lhcInfo) { setFromLHCInfo(lhcInfo); }

LHCInfoCombined::LHCInfoCombined(const LHCInfoPerLS& infoPerLS, const LHCInfoPerFill& infoPerFill) {
  setFromPerLS(infoPerLS);
  setFromPerFill(infoPerFill);
}

LHCInfoCombined::LHCInfoCombined(const edm::EventSetup& iSetup,
                                 const edm::ESGetToken<LHCInfoPerLS, LHCInfoPerLSRcd>& tokenInfoPerLS,
                                 const edm::ESGetToken<LHCInfoPerFill, LHCInfoPerFillRcd>& tokenInfoPerFill,
                                 const edm::ESGetToken<LHCInfo, LHCInfoRcd>& tokenInfo,
                                 bool useNewLHCInfo) {
  if (useNewLHCInfo) {
    edm::ESHandle<LHCInfoPerLS> hLHCInfoPerLS = iSetup.getHandle(tokenInfoPerLS);
    edm::ESHandle<LHCInfoPerFill> hLHCInfoFill = iSetup.getHandle(tokenInfoPerFill);
    setFromPerLS(*hLHCInfoPerLS);
    setFromPerFill(*hLHCInfoFill);
  } else {
    edm::ESHandle<LHCInfo> hLHCInfo = iSetup.getHandle(tokenInfo);
    setFromLHCInfo(*hLHCInfo);
  }
}

void LHCInfoCombined::setFromLHCInfo(const LHCInfo& lhcInfo) {
  crossingAngleX = lhcInfo.crossingAngle();
  crossingAngleY = 0;
  betaStarX = lhcInfo.betaStar();
  betaStarY = lhcInfo.betaStar();
  energy = lhcInfo.energy();
  fillNumber = lhcInfo.fillNumber();
}
void LHCInfoCombined::setFromPerLS(const LHCInfoPerLS& infoPerLS) {
  crossingAngleX = infoPerLS.crossingAngleX();
  crossingAngleY = infoPerLS.crossingAngleY();
  betaStarX = infoPerLS.betaStarX();
  betaStarY = infoPerLS.betaStarY();
}
void LHCInfoCombined::setFromPerFill(const LHCInfoPerFill& infoPerFill) {
  energy = infoPerFill.energy();
  fillNumber = infoPerFill.fillNumber();
}

float LHCInfoCombined::crossingAngle() {
  if (crossingAngleX == 0. && crossingAngleY == 0.) {
    return crossingAngleInvalid;
  }
  if (crossingAngleX != 0. && crossingAngleY != 0.) {
    edm::LogWarning("LHCInfoCombined") << "crossingAngleX and crossingAngleY are both different from 0";
    return crossingAngleInvalid;
  }
  return crossingAngleX == 0. ? crossingAngleY : crossingAngleX;
}

//Comparison with the -1 value from LHC when crossing angle is not set
bool LHCInfoCombined::isCrossingAngleInvalid() {
  float comparisonTolerance = 1e-6;
  return fabs(crossingAngle() - crossingAngleInvalid) <= comparisonTolerance;
}

void LHCInfoCombined::print(std::ostream& os) const {
  os << "Crossing angle x (urad): " << crossingAngleX << std::endl
     << "Crossing angle y (urad): " << crossingAngleY << std::endl
     << "Beta star x (m): " << betaStarX << std::endl
     << "Beta star y (m): " << betaStarY << std::endl
     << "Energy (GeV): " << energy << std::endl;
}

std::ostream& operator<<(std::ostream& os, LHCInfoCombined beamInfo) {
  beamInfo.print(os);
  return os;
}
