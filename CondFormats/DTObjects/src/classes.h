/* Condtion Objects
 * DTReadOutMapping
 * DTT0
 * DTT0
 * DTRangeT0
 * DTTtrig
 * DTMtime
 * DTStatusFlag
 * DTDeadFlag
 * DTPerformance
 * DTCCBConfig
 * DTTPGParameters
 */

#include "CondFormats/Common/interface/PayloadWrapper.h"

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
#include "CondFormats/DTObjects/interface/DTTPGParameters.h"

namespace {
  struct dictionary {
    std::pair<              int,      DTConfigToken>   confTokenPair;
    std::pair<           DTT0Id,           DTT0Data>          t0Pair;
    std::pair<        DTTtrigId,        DTTtrigData>       tTrigPair;
    std::pair<        DTMtimeId,        DTMtimeData>       mTimePair;
    std::pair<      DTRangeT0Id,      DTRangeT0Data>     rangeT0Pair;
    std::pair<   DTStatusFlagId,   DTStatusFlagData>  statusFlagPair;
    std::pair<     DTDeadFlagId,     DTDeadFlagData>    deadFlagPair;
    std::pair<  DTPerformanceId,  DTPerformanceData> performancePair;
    std::pair<          DTCCBId,                int>         ccbPair;
    std::pair<DTTPGParametersId,DTTPGParametersData>         tpgPair;

    std::vector< DTReadOutGeometryLink >            readoutMap;
    std::vector< std::pair<              int,
                               DTConfigToken> >   confTokenMap;
    std::vector< std::pair<           DTT0Id,
                                    DTT0Data> >          t0Map;
    std::vector< std::pair<        DTTtrigId,
                                 DTTtrigData> >       tTrigMap;
    std::vector< std::pair<        DTMtimeId,
                                 DTMtimeData> >       mTimeMap;
    std::vector< std::pair<      DTRangeT0Id,
                               DTRangeT0Data> >     rangeT0Map;
    std::vector< std::pair<   DTStatusFlagId,
                            DTStatusFlagData> >  statusFlagMap;
    std::vector< std::pair<     DTDeadFlagId,
                              DTDeadFlagData> >    deadFlagMap;
    std::vector< std::pair<  DTPerformanceId,
                           DTPerformanceData> > performanceMap;
    std::vector< std::pair<          DTCCBId,
                                         int> >         ccbMap;
    std::vector< std::pair<DTTPGParametersId,
                         DTTPGParametersData> >         tpgMap;



//    std::vector< std::pair<   DTT0Id,DTT0Data> > blah2;
//    std::vector< std::pair<DTTtrigId,DTTtrigData> > blah3;
//    std::vector< std::pair<DTMtimeId,DTMtimeData> > blah4;
//    std::vector< std::pair<DTStatusFlagId,DTStatusFlagData> > blah5;
  };
}


