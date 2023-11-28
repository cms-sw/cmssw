#ifndef CondTools_RunInfo_LHCInfoCombined_H
#define CondTools_RunInfo_LHCInfoCombined_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/RunInfo/interface/LHCInfoPerLS.h"
#include "CondFormats/RunInfo/interface/LHCInfoPerFill.h"

#include "CondFormats/DataRecord/interface/LHCInfoPerLSRcd.h"
#include "CondFormats/DataRecord/interface/LHCInfoPerFillRcd.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

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
                  const edm::ESGetToken<LHCInfo, LHCInfoRcd>& tokenInfo,
                  bool useNewLHCInfo);

  //this factory method is necessary because constructor can't be a template
  template <class RecordT, class ListT>
  static LHCInfoCombined createLHCInfoCombined(
      const edm::eventsetup::DependentRecordImplementation<RecordT, ListT>& iRecord,
      const edm::ESGetToken<LHCInfoPerLS, LHCInfoPerLSRcd>& tokenInfoPerLS,
      const edm::ESGetToken<LHCInfoPerFill, LHCInfoPerFillRcd>& tokenInfoPerFill,
      const edm::ESGetToken<LHCInfo, LHCInfoRcd>& tokenInfo,
      bool useNewLHCInfo);

  void setFromLHCInfo(const LHCInfo& lhcInfo);
  void setFromPerLS(const LHCInfoPerLS& infoPerLS);
  void setFromPerFill(const LHCInfoPerFill& infoPerFill);

  float crossingAngle();
  static constexpr float crossingAngleInvalid = -1.;
  bool isCrossingAngleInvalid();

  float crossingAngleX;
  float crossingAngleY;
  float betaStarX;
  float betaStarY;
  float energy;
  unsigned short fillNumber;

  void print(std::ostream& os) const;
};

std::ostream& operator<<(std::ostream& os, LHCInfoCombined beamInfo);

template <class RecordT, class ListT>
LHCInfoCombined LHCInfoCombined::createLHCInfoCombined(
    const edm::eventsetup::DependentRecordImplementation<RecordT, ListT>& iRecord,
    const edm::ESGetToken<LHCInfoPerLS, LHCInfoPerLSRcd>& tokenInfoPerLS,
    const edm::ESGetToken<LHCInfoPerFill, LHCInfoPerFillRcd>& tokenInfoPerFill,
    const edm::ESGetToken<LHCInfo, LHCInfoRcd>& tokenInfo,
    bool useNewLHCInfo) {
  LHCInfoCombined lhcInfoCombined;
  if (useNewLHCInfo) {
    LHCInfoPerLS const& lhcInfoPerLS = iRecord.get(tokenInfoPerLS);
    LHCInfoPerFill const& lhcInfoPerFill = iRecord.get(tokenInfoPerFill);
    lhcInfoCombined.setFromPerLS(lhcInfoPerLS);
    lhcInfoCombined.setFromPerFill(lhcInfoPerFill);
  } else {
    LHCInfo const& lhcInfo = iRecord.get(tokenInfo);
    lhcInfoCombined.setFromLHCInfo(lhcInfo);
  }
  return lhcInfoCombined;
}

#endif  // CondTools_RunInfo_LHCInfoCombined_H
