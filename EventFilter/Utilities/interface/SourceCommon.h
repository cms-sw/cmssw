#ifndef EventFilter_Utilities_SourceCommon_h
#define EventFilter_Utilities_SourceCommon_h

/*
 * This header will host common definitions used by FedRawDataInputSource and DAQSource
 * */

#include "EventFilter/Utilities/interface/FastMonitoringService.h"

class IdleSourceSentry {
public:
  IdleSourceSentry(evf::FastMonitoringService* fms) : fms_(fms) {
    if (fms_)
      fms_->setTMicrostate(evf::FastMonState::mIdleSource);
  }
  ~IdleSourceSentry() {
    if (fms_)
      fms_->setTMicrostate(evf::FastMonState::mIdle);
  }

private:
  evf::FastMonitoringService* fms_;
};

#endif
