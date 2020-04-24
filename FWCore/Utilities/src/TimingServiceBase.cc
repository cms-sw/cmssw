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
double TimingServiceBase::s_jobStartTime = 0.0;

void TimingServiceBase::jobStarted() {
  if (0.0 == s_jobStartTime) {
    struct timeval t;
    if(gettimeofday(&t, nullptr) < 0) {
      return;
    }
    s_jobStartTime = static_cast<double>(t.tv_sec) + (static_cast<double>(t.tv_usec) * 1E-6);
  }
}

//
// constructors and destructor
//
TimingServiceBase::TimingServiceBase()
{
}

TimingServiceBase::~TimingServiceBase()
{
}
