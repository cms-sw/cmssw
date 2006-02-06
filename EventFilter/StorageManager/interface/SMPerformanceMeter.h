#ifndef EVFSM_SMPerformanceMeter_h_
#define EVFSM_SMPerformanceMeter_h_
/*
   Author: Harry Cheung, FNAL

   Description:
     Header file for performance statistics for
     Storage Manager and I2O output module.
       This is a modified version of code taken from
     the XDAQ toolbox::PerformanceMeter written by
     J. Gutleber and L. Orsini

   Modification:
     version 1.1 2005/12/15
       Initial implementation. Modified from toolbox 
       version to keep a running mean, and max and min values.

*/

#include <string>

#include "toolbox/Chrono.h"
#include "toolbox/string.h"

namespace sto {

  class SMPerformanceMeter 
  {
    public:

    SMPerformanceMeter();

    virtual ~SMPerformanceMeter(){}

    // Take the received amount of data in bytes and
    // add it to the total amount of received bytes.
    //   Update the statistics after addSample
    // is called samples_ times. Keep a runnning mean.
    // Return true if set of samples is reached.

    void init(unsigned long samples);
    bool addSample(unsigned long size);
  
    double bandwidth();
    double rate();
    double latency();
    double meanbandwidth();
    double meanrate();
    double meanlatency();
    unsigned long totalsamples();
    double duration();

    protected:

    // variables for the mean over the whole run
    double totalMB4mean_;
    double meanThroughput_;
    double meanRate_;
    double  meanLatency_;
    unsigned long sampleCounter_;
    double  allTime_;
    // variables for each set of "samples_"
    double totalMB_;
    double throughput_;
    double rate_;
    double  latency_;
    unsigned long loopCounter_;
    unsigned long samples_;
    toolbox::Chrono chrono_;	
  }; //end class

} // end namespace sto

#endif
