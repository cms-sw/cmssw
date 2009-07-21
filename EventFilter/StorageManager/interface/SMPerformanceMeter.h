#ifndef EVFSM_SMPerformanceMeter_h_
#define EVFSM_SMPerformanceMeter_h_
/*
     Header file for performance statistics for
     Storage Manager and SMProxyServer.

     $Id: SMPerformanceMeter.h,v 1.7 2009/06/10 08:15:23 dshpakov Exp $
*/

#include <string>

#include "toolbox/string.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"
#include "EventFilter/StorageManager/interface/ForeverCounter.h"
#include "EventFilter/StorageManager/interface/RollingSampleCounter.h"
#include "EventFilter/StorageManager/interface/RollingIntervalCounter.h"

namespace stor {

  struct SMPerfStats
  {
    SMPerfStats();
    public:
    void reset();
    void fullReset();
    unsigned long samples_;
    unsigned long period4samples_;
    boost::shared_ptr<ForeverCounter> longTermCounter_;
    boost::shared_ptr<RollingSampleCounter> shortTermCounter_;
    boost::shared_ptr<RollingIntervalCounter> shortPeriodCounter_;
    // for sample based statistics
    double maxBandwidth_;
    double minBandwidth_;
    // for time period based statistics
    double maxBandwidth2_;
    double minBandwidth2_;
  };

  struct SMOnlyStats
  {
    SMOnlyStats();
    public:
    unsigned long samples_;
    unsigned long period4samples_;
    // for sample based statistics
    double instantBandwidth_;
    double instantRate_;
    double instantLatency_;
    double totalSamples_;
    double duration_;
    double meanBandwidth_;
    double meanRate_;
    double meanLatency_;
    double maxBandwidth_;
    double minBandwidth_;

    // for time period based statistics
    double instantBandwidth2_;
    double instantRate2_;
    double instantLatency2_;
    double totalSamples2_;
    double duration2_;
    double meanBandwidth2_;
    double meanRate2_;
    double meanLatency2_;
    double maxBandwidth2_;
    double minBandwidth2_;

    double receivedVolume_;
  };

  class SMPerformanceMeter 
  {
    public:

    SMPerformanceMeter();

    virtual ~SMPerformanceMeter(){}

    void init(unsigned long samples, unsigned long time_period);
    bool addSample(unsigned long size);
    void setSamples(unsigned long num_samples);
    void setPeriod4Samples(unsigned long time_period);
    unsigned long getSetSamples() { return stats_.samples_; }
    unsigned long getPeriod4Samples() { return stats_.period4samples_; }
  
    SMPerfStats getStats();
    unsigned long samples();
    double totalvolumemb();

    protected:

    unsigned long loopCounter_;
    SMPerfStats stats_;

    boost::mutex data_lock_;
  }; //end class

} // end namespace stor

#endif
/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
