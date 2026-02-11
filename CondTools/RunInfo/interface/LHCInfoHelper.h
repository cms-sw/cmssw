#ifndef CondTools_RunInfo_LHCInfoHelper_h
#define CondTools_RunInfo_LHCInfoHelper_h

#include "CondCore/CondDB/interface/Time.h"
#include "CondTools/RunInfo/interface/OMSAccess.h"

namespace cond {

  namespace lhcInfoHelper {

    // Large number of LS for the OMS query, covering around 25 hours
    static constexpr unsigned int kLumisectionsQueryLimit = 4000;

    // last Run number and LS number of the specified Fill
    std::pair<int, unsigned short> getFillLastRunAndLS(const cond::OMSService& oms, unsigned short fillId);

    // Returns lumi-type IOV from last LS of last Run of the specified Fill
    cond::Time_t getFillLastLumiIOV(const cond::OMSService& oms, unsigned short fillId);

  }  // namespace lhcInfoHelper

}  // namespace cond

#endif