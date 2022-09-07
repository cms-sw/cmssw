#ifndef CondFormats_RunInfo_LHCInfoCombined_H
#define CondFormats_RunInfo_LHCInfoCombined_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/RunInfo/interface/LHCInfoPerLS.h"
#include "CondFormats/RunInfo/interface/LHCInfoPerFill.h"

#include "CondFormats/DataRecord/interface/LHCInfoPerLSRcd.h"
#include "CondFormats/DataRecord/interface/LHCInfoPerFillRcd.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "CondCore/CondDB/interface/Types.h"

#include <bitset>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

class LHCInfoCombined {
public:
  LHCInfoCombined() = default;

  LHCInfoCombined(const LHCInfo& lhcInfo);
  LHCInfoCombined(const LHCInfoPerLS& infoPerLS, const LHCInfoPerFill& infoPerFill);
  LHCInfoCombined(const edm::EventSetup& iSetup,
                  const edm::ESGetToken<LHCInfoPerLS, LHCInfoPerLSRcd>& tokenInfoPerLS,
                  const edm::ESGetToken<LHCInfoPerFill, LHCInfoPerFillRcd>& tokenInfoPerFill,
                  const edm::ESGetToken<LHCInfo, LHCInfoRcd>& tokenInfo);

  void setFromLHCInfo(const LHCInfo& lhcInfo);
  void setFromPerLS(const LHCInfoPerLS& infoPerLS);
  void setFromPerFill(const LHCInfoPerFill& infoPerFill);

  float crossingAngleX;
  float crossingAngleY;
  float betaStarX;
  float betaStarY;
  float energy;

  void print(std::stringstream& ss) const;
};

std::ostream& operator<<(std::ostream& os, LHCInfoCombined beamInfo);

#endif  // CondFormats_RunInfo_LHCInfoCombined_H
