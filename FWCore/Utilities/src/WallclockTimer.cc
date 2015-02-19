// -*- C++ -*-
//
// Package:     Utilities
// Class  :     WallclockTimer
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sun Apr 16 20:32:20 EDT 2006
//

// system include files
#include <sys/resource.h>
#include <errno.h>

// user include files
#include "FWCore/Utilities/interface/WallclockTimer.h"
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
WallclockTimer::WallclockTimer() :
state_(kStopped),
startRealTime_(),
accumulatedRealTime_(0)
{
#ifdef USE_CLOCK_GETTIME
  startRealTime_.tv_sec=0;
  startRealTime_.tv_nsec=0;
#else
  startRealTime_.tv_sec=0;
  startRealTime_.tv_usec=0;
#endif
}

// WallclockTimer::WallclockTimer(WallclockTimer const& rhs) {
//    // do actual copying here;
// }

WallclockTimer::~WallclockTimer() {
}

//
// assignment operators
//
// WallclockTimer const& WallclockTimer::operator=(WallclockTimer const& rhs) {
//   //An exception safe implementation is
//   WallclockTimer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
WallclockTimer::start() {
  if(kStopped == state_) {
#ifdef USE_CLOCK_GETTIME
    clock_gettime(CLOCK_MONOTONIC, &startRealTime_);
#else
    gettimeofday(&startRealTime_, 0);
#endif
    state_ = kRunning;
  }
}

double
WallclockTimer::stop() {
  if(kRunning == state_) {
    auto t = calculateDeltaTime();
    accumulatedRealTime_ += t;

    state_=kStopped;
    return t;
  }
  return 0.;
}

void
WallclockTimer::reset() {
  accumulatedRealTime_ = 0;
}

void
WallclockTimer::add(double t) {
  accumulatedRealTime_ += t;
}

double
WallclockTimer::calculateDeltaTime() const {
  double returnValue;
#ifdef USE_CLOCK_GETTIME
  double const nanosecToSec = 1E-9;
  struct timespec tp;

  clock_gettime(CLOCK_MONOTONIC, &tp);
  returnValue = tp.tv_sec - startRealTime_.tv_sec + nanosecToSec * (tp.tv_nsec - startRealTime_.tv_nsec);
#else
  double const microsecToSec = 1E-6;

  struct timeval tp;
  gettimeofday(&tp, 0);

  returnValue = tp.tv_sec - startRealTime_.tv_sec + microsecToSec * (tp.tv_usec - startRealTime_.tv_usec);
#endif
  return returnValue;
}
//
// const member functions
//
double
WallclockTimer::realTime() const {
  if(kStopped == state_) {
    return accumulatedRealTime_;
  }
  return accumulatedRealTime_ + calculateDeltaTime();
}

//
// static member functions
//
