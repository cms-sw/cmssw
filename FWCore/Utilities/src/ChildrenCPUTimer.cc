// -*- C++ -*-
//
// Package:     Utilities
// Class  :     ChildrenCPUTimer
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sun Apr 16 20:32:20 EDT 2006
//

// system include files
#include <sys/resource.h>
#include <cerrno>

// user include files
#include "FWCore/Utilities/interface/ChildrenCPUTimer.h"
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
ChildrenCPUTimer::ChildrenCPUTimer() : state_(kStopped), startCPUTime_(), accumulatedCPUTime_(0) {
  startCPUTime_.tv_sec = 0;
  startCPUTime_.tv_usec = 0;
}

ChildrenCPUTimer::~ChildrenCPUTimer() {}

//
// member functions
//
void ChildrenCPUTimer::start() {
  if (kStopped == state_) {
    rusage theUsage;
    if (0 != getrusage(RUSAGE_CHILDREN, &theUsage)) {
      throw cms::Exception("ChildrenCPUTimerFailed") << errno;
    }
    startCPUTime_.tv_sec = theUsage.ru_stime.tv_sec + theUsage.ru_utime.tv_sec;
    startCPUTime_.tv_usec = theUsage.ru_stime.tv_usec + theUsage.ru_utime.tv_usec;
    state_ = kRunning;
  }
}

double ChildrenCPUTimer::stop() {
  if (kRunning == state_) {
    auto t = calculateDeltaTime();
    accumulatedCPUTime_ += t;

    state_ = kStopped;
    return t;
  }
  return 0.;
}

void ChildrenCPUTimer::reset() { accumulatedCPUTime_ = 0; }

void ChildrenCPUTimer::add(double t) { accumulatedCPUTime_ += t; }

double ChildrenCPUTimer::calculateDeltaTime() const {
  double returnValue;
  double const microsecToSec = 1E-6;

  rusage theUsage;
  if (0 != getrusage(RUSAGE_CHILDREN, &theUsage)) {
    throw cms::Exception("CPUTimerFailed") << errno;
  }

  returnValue = theUsage.ru_stime.tv_sec + theUsage.ru_utime.tv_sec - startCPUTime_.tv_sec +
                microsecToSec * (theUsage.ru_stime.tv_usec + theUsage.ru_utime.tv_usec - startCPUTime_.tv_usec);
  return returnValue;
}
//
// const member functions
//
double ChildrenCPUTimer::cpuTime() const {
  if (kStopped == state_) {
    return accumulatedCPUTime_;
  }
  return accumulatedCPUTime_ + calculateDeltaTime();
}

//
// static member functions
//
