/* Condition Objects
 * DTReadOutMapping
 * DTT0
 * DTRangeT0
 * DTTtrig
 * DTMtime
 * DTStatusFlag
 * DTDeadFlag
 * DTPerformance
 * DTLVStatus
 * DTHVStatus
 * DTCCBConfig
 * DTTPGParameters
 */

#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DTObjects/interface/DTRangeT0.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"
#include "CondFormats/DTObjects/interface/DTPerformance.h"
#include "CondFormats/DTObjects/interface/DTLVStatus.h"
#include "CondFormats/DTObjects/interface/DTHVStatus.h"
#include "CondFormats/DTObjects/interface/DTCCBConfig.h"
#include "CondFormats/DTObjects/interface/DTKeyedConfig.h"
#include "CondFormats/DTObjects/interface/DTTPGParameters.h"
#include "CondFormats/DTObjects/interface/DTRecoUncertainties.h"

namespace CondFormats_DTObjects {
  struct dictionary {
//    std::pair<           DTT0Id,           DTT0Data>          t0Pair;
    std::pair<        DTTtrigId,        DTTtrigData>       tTrigPair;
    std::pair<        DTMtimeId,        DTMtimeData>       mTimePair;
    std::pair<      DTRangeT0Id,      DTRangeT0Data>     rangeT0Pair;
    std::pair<   DTStatusFlagId,   DTStatusFlagData>  statusFlagPair;
    std::pair<     DTDeadFlagId,     DTDeadFlagData>    deadFlagPair;
    std::pair<  DTPerformanceId,  DTPerformanceData> performancePair;
    std::pair<     DTLVStatusId,     DTLVStatusData>    lvStatusPair;
    std::pair<     DTHVStatusId,     DTHVStatusData>    hvStatusPair;
    std::pair<          DTCCBId,                int>         ccbPair;
    std::pair<DTTPGParametersId,DTTPGParametersData>         tpgPair;

    std::vector< DTReadOutGeometryLink >            readoutMap;
//    std::vector< std::pair<           DTT0Id,
//                                    DTT0Data> >          t0Map;
    std::vector< DTT0Data                     >          t0Map;
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
    std::vector< std::pair<     DTLVStatusId,
                              DTLVStatusData> >    lvStatusMap;
    std::vector< std::pair<     DTHVStatusId,
                              DTHVStatusData> >    hvStatusMap;
    std::vector<                 DTConfigKey  >        confKey;
    std::vector< std::pair<          DTCCBId,
                                         int> >         ccbMap;
    std::vector< std::pair<DTTPGParametersId,
                         DTTPGParametersData> >         tpgMap;



//    std::vector< std::pair<   DTT0Id,DTT0Data> > blah2;
//    std::vector< std::pair<DTTtrigId,DTTtrigData> > blah3;
//    std::vector< std::pair<DTMtimeId,DTMtimeData> > blah4;
//    std::vector< std::pair<DTStatusFlagId,DTStatusFlagData> > blah5;

    
    std::pair<uint32_t, std::vector<float> > p_payload;
    std::map<uint32_t, std::vector<float> > payload;

  };
}


/*
// wrapper declarations
namespace CondFormats_DTObjects {
   struct wrappers {
      pool::Ptr<DTReadOutMapping >          pMap;
      cond::DataWrapper<DTReadOutMapping > dwMap;
      pool::Ptr<DTT0 >                     pT0;
      cond::DataWrapper<DTT0 >            dwT0;
      pool::Ptr<DTRangeT0 >                pRangeT0;
      cond::DataWrapper<DTRangeT0 >       dwRangeT0;
      pool::Ptr<DTTtrig >                  pTtrig;
      cond::DataWrapper<DTTtrig >         dwTtrig;
      pool::Ptr<DTMtime >                  pMTime;
      cond::DataWrapper<DTMtime >         dwMTime;
      pool::Ptr<DTStatusFlag >             pStatusFlag;
      cond::DataWrapper<DTStatusFlag >    dwStatusFlag;
      pool::Ptr<DTDeadFlag >               pDeadFlag;
      cond::DataWrapper<DTDeadFlag >      dwDeadFlag;
      pool::Ptr<DTPerformance >            pPerformance;
      cond::DataWrapper<DTPerformance >   dwPerformance;
      pool::Ptr<DTCCBConfig >              pCCBConfig;
      cond::DataWrapper<DTCCBConfig >     dwCCBConfig;
      pool::Ptr<DTLVStatus >               pLVStatus;
      cond::DataWrapper<DTLVStatus >      dwLVStatus;
      pool::Ptr<DTHVStatus >               pHVStatus;
      cond::DataWrapper<DTHVStatus >      dwHVStatus;
      pool::Ptr<DTTPGParameters >          pTPGParameter;
      cond::DataWrapper<DTTPGParameters > dwTPGParameter;

   };
}
*/
