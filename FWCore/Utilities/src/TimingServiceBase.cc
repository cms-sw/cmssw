// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     TimingServiceBase
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Wed, 11 Jun 2014 15:08:00 GMT
//

// system include files
#include <sys/resource.h>
#include <sys/time.h>

// user include files
#include "FWCore/Utilities/interface/TimingServiceBase.h"

using namespace edm;
//
// constants, enums and typedefs
//
std::chrono::steady_clock::time_point TimingServiceBase::s_jobStartTime;

void TimingServiceBase::jobStarted() {
  if (0 == s_jobStartTime.time_since_epoch().count()) {
    s_jobStartTime = std::chrono::steady_clock::now();
  }
}

//
// constructors and destructor
//
TimingServiceBase::TimingServiceBase() {}

TimingServiceBase::~TimingServiceBase() {}
