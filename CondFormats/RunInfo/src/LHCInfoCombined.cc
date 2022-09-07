#include "CondFormats/RunInfo/interface/LHCInfoCombined.h"

LHCInfoCombined::LHCInfoCombined(const LHCInfo& lhcInfo) { setFromLHCInfo(lhcInfo); }

LHCInfoCombined::LHCInfoCombined(const LHCInfoPerLS& infoPerLS, const LHCInfoPerFill& infoPerFill) {
  setFromPerLS(infoPerLS);
  setFromPerFill(infoPerFill);
}

LHCInfoCombined::LHCInfoCombined(const edm::EventSetup& iSetup,
                                 const edm::ESGetToken<LHCInfoPerLS, LHCInfoPerLSRcd>& tokenInfoPerLS,
                                 const edm::ESGetToken<LHCInfoPerFill, LHCInfoPerFillRcd>& tokenInfoPerFill,
                                 const edm::ESGetToken<LHCInfo, LHCInfoRcd>& tokenInfo) {
  if (true /* era run3 */) {
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
}
void LHCInfoCombined::setFromPerLS(const LHCInfoPerLS& infoPerLS) {
  crossingAngleX = infoPerLS.crossingAngleX();
  crossingAngleY = infoPerLS.crossingAngleY();
  betaStarX = infoPerLS.betaStarX();
  betaStarY = infoPerLS.betaStarY();
}
void LHCInfoCombined::setFromPerFill(const LHCInfoPerFill& infoPerFill) { energy = infoPerFill.energy(); }

void LHCInfoCombined::print(std::stringstream& ss) const {
  ss << "Crossing angle x (urad): " << crossingAngleX << std::endl
     << "Crossing angle y (urad): " << crossingAngleY << std::endl
     << "Beta star x (m): " << betaStarX << std::endl
     << "Beta star y (m): " << betaStarY << std::endl
     << "Energy (GeV): " << energy << std::endl;
}

std::ostream& operator<<(std::ostream& os, LHCInfoCombined beamInfo) {
  std::stringstream ss;
  beamInfo.print(ss);
  os << ss.str();
  return os;
}
