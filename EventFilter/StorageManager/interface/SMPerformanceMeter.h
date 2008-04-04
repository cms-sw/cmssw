#ifndef EVFSM_SMPerformanceMeter_h_
#define EVFSM_SMPerformanceMeter_h_
/*
     Header file for performance statistics for
     Storage Manager and SMProxyServer.

     $Id$
*/

#include <string>

#include "toolbox/Chrono.h"
#include "toolbox/string.h"

#include "boost/thread/thread.hpp"

namespace stor {

  struct SMPerfStats
  {
    SMPerfStats();
    public:
    void reset();
    void fullReset();
    // variables for the mean over the whole run
    unsigned long samples_;
    double totalMB4mean_;
    double meanThroughput_;
    double meanRate_;
    double  meanLatency_;
    unsigned long sampleCounter_;
    double  allTime_;
    double maxBandwidth_;
    double minBandwidth_;
    // variables for each set of "samples_"
    double totalMB_;
    double throughput_;
    double rate_;
    double  latency_;
  };

  class SMPerformanceMeter 
  {
    public:

    SMPerformanceMeter();

    virtual ~SMPerformanceMeter(){}

    void init(unsigned long samples);
    bool addSample(unsigned long size);
    void setSamples(unsigned long num_samples);
  
    SMPerfStats getStats();
    unsigned long samples();
    double bandwidth();
    double rate();
    double latency();
    double meanbandwidth();
    double maxbandwidth();
    double minbandwidth();
    double meanrate();
    double meanlatency();
    unsigned long totalsamples();
    double totalvolumemb();
    double duration();

    protected:

    SMPerfStats stats_;
    unsigned long loopCounter_;
    toolbox::Chrono chrono_;	

    boost::mutex data_lock_;
  }; //end class

} // end namespace stor

#endif
