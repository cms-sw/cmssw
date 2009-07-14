#ifndef COND_DBCommon_ConvertIOVSyncValue_H
#define COND_DBCommon__ConvertIOVSyncValue_H

#include "CondCore/DBCommon/interface/Time.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"

namespace cond {

  edm::IOVSyncValue toIOVSyncValue(cond::Time_t time, cond::TimeType timetype);
  cond::Time_t fromIOVSyncValue(edm::IOVSyncValue const & time, cond::TimeType timetype);


}


#enddo
