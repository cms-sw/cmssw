#ifndef COND_DBCommon_ConvertIOVSyncValue_H
#define COND_DBCommon__ConvertIOVSyncValue_H

#include "CondCore/DBCommon/interface/Time.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"

namespace cond {

  edm::IOVSyncValue toIOVSyncValue(cond::Time_t time, cond::TimeType timetype, bool startOrStop);

  cond::Time_t fromIOVSyncValue(edm::IOVSyncValue const & time, cond::TimeType timetype);

  // min max sync value.... 
  edm::IOVSyncValue limitedIOVSyncValue(cond::Time_t time, cond::TimeType timetype);

  edm::IOVSyncValue limitedIOVSyncValue(edm::IOVSyncValue const & time, cond::TimeType timetype);


}


#endif
