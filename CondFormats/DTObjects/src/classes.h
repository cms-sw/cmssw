#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DTObjects/interface/DTRangeT0.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"
#include "CondFormats/DTObjects/interface/DTPerformance.h"
#include "CondFormats/DTObjects/interface/DTCCBConfig.h"
#include "CondFormats/DTObjects/interface/DTConfigList.h"
#include "CondFormats/DTObjects/interface/DTConfigData.h"

namespace {
  struct dictionary {
    std::pair<int,DTConfigToken> refMap;
    std::pair<DTT0Id,DTT0Data> t0Pair;
    std::pair<DTTtrigId,DTTtrigData> tTrigPair;
    std::pair<DTMtimeId,DTMtimeData> mTimePair;
    std::pair<DTRangeT0Id,DTRangeT0Data> rangeT0Pair;
    std::pair<DTStatusFlagId,DTStatusFlagData> statusFlagPair;
    std::pair<DTDeadFlagId,DTDeadFlagData> deadFlagPair;
    std::pair<DTPerformanceId,DTPerformanceData> performancePair;
    std::pair<DTCCBId,int> ccbPair;
 
    std::vector<DTReadOutGeometryLink> blah1;
    std::vector<std::pair<DTT0Id,DTT0Data> > blah2;
    std::vector< std::pair<DTTtrigId,DTTtrigData> > blah3;
    std::vector< std::pair<DTMtimeId,DTMtimeData> > blah4;
    std::vector< std::pair<DTStatusFlagId,DTStatusFlagData> > blah5;
  };
}


