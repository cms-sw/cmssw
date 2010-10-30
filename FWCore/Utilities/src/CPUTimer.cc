// -*- C++ -*-
//
// Package:     Utilities
// Class  :     CPUTimer
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sun Apr 16 20:32:20 EDT 2006
// $Id: CPUTimer.cc,v 1.3 2007/06/14 02:01:01 wmtan Exp $
//

// system include files
#include <sys/resource.h>
#include <errno.h>

// user include files
#include "FWCore/Utilities/interface/CPUTimer.h"
#include "FWCore/Utilities/interface/Exception.h"

//
// constants, enums and typedefs
//
using namespace edm;

//
// static data member definitions
//

//
// constructors and destructor
//
CPUTimer::CPUTimer() :
state_(kStopped),
startRealTime_(),
startCPUTime_(),
accumulatedRealTime_(0),
accumulatedCPUTime_(0)
{
#ifdef USE_CLOCK_GETTIME
  startRealTime_.tv_sec=0;
  startRealTime_.tv_nsec=0;
  startCPUTime_.tv_sec=0;
  startCPUTime_.tv_nsec=0;
#else
  startRealTime_.tv_sec=0;
  startRealTime_.tv_usec=0;
  startCPUTime_.tv_sec=0;
  startCPUTime_.tv_usec=0;
#endif
}

// CPUTimer::CPUTimer(const CPUTimer& rhs)
// {
//    // do actual copying here;
// }

CPUTimer::~CPUTimer()
{
}

//
// assignment operators
//
// const CPUTimer& CPUTimer::operator=(const CPUTimer& rhs)
// {
//   //An exception safe implementation is
//   CPUTimer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
CPUTimer::start() {
  if(kStopped == state_) {
#ifdef USE_CLOCK_GETTIME
    clock_gettime(CLOCK_REALTIME,&startRealTime_);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&startCPUTime_);
#else
    rusage theUsage;
    if( 0 != getrusage(RUSAGE_SELF, &theUsage)) {
      throw cms::Exception("CPUTimerFailed")<<errno;
    }
    startCPUTime_.tv_sec =theUsage.ru_stime.tv_sec+theUsage.ru_utime.tv_sec;
    startCPUTime_.tv_usec =theUsage.ru_stime.tv_usec+theUsage.ru_utime.tv_usec;
    
    gettimeofday(&startRealTime_, 0);
#endif
    state_ = kRunning;
  }
}

CPUTimer::Times 
CPUTimer::stop() {
  if(kRunning == state_) {
    Times t = calculateDeltaTime();
    accumulatedCPUTime_ += t.cpu_;
    accumulatedRealTime_ += t.real_;

    state_=kStopped;
    return t;
  }
  return Times();
}

void 
CPUTimer::reset(){
  accumulatedCPUTime_ =0;
  accumulatedRealTime_=0;
}

void 
CPUTimer::add(const CPUTimer::Times& t)
{
  accumulatedCPUTime_ += t.cpu_;
  accumulatedRealTime_ += t.real_;  
}



CPUTimer::Times
CPUTimer::calculateDeltaTime() const
{
  Times returnValue;
#ifdef USE_CLOCK_GETTIME
  const double nanosecToSec = 1E-9;
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME,&tp);
  returnValue.real_ = tp.tv_sec-startRealTime_.tv_sec+nanosecToSec*(tp.tv_nsec-startRealTime_.tv_nsec);
  
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&tp);
  returnValue.cpu_ = tp.tv_sec-startCPUTime_.tv_sec+nanosecToSec*(tp.tv_nsec-startCPUTime_.tv_nsec);
#else
  rusage theUsage;
  if( 0 != getrusage(RUSAGE_SELF, &theUsage)) {
    throw cms::Exception("CPUTimerFailed")<<errno;
  }
  const double microsecToSec = 1E-6;
  
  struct timeval tp;
  gettimeofday(&tp, 0);
  
  returnValue.cpu_ = theUsage.ru_stime.tv_sec+theUsage.ru_utime.tv_sec-startCPUTime_.tv_sec+microsecToSec*(theUsage.ru_stime.tv_usec+theUsage.ru_utime.tv_usec-startCPUTime_.tv_usec);
  returnValue.real_ = tp.tv_sec-startRealTime_.tv_sec+microsecToSec*(tp.tv_usec -startRealTime_.tv_usec);
#endif
  return returnValue;
}
//
// const member functions
//
double 
CPUTimer::realTime() const 
{ 
  if(kStopped == state_) {
    return accumulatedRealTime_;
  }
  return accumulatedRealTime_ + calculateDeltaTime().real_; 
}

double 
CPUTimer::cpuTime() const 
{ 
  if(kStopped== state_) {
    return accumulatedCPUTime_;
  }
  return accumulatedCPUTime_+ calculateDeltaTime().cpu_;
}

//
// static member functions
//
