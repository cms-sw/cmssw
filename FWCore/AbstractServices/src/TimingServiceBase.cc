// -*- C++ -*-
//
// Package:     FWCore/AbstractServices
// Class  :     TimingServiceBase
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Wed, 11 Jun 2014 15:08:00 GMT
//

#include "FWCore/AbstractServices/interface/TimingServiceBase.h"

using namespace edm;

void TimingServiceBase::jobStarted() {
  //make sure the value has been initialized
  (void)jobStartTime();
}

std::chrono::steady_clock::time_point TimingServiceBase::jobStartTime() {
  static const std::chrono::steady_clock::time_point s_jobStartTime = std::chrono::steady_clock::now();
  return s_jobStartTime;
}

void TimingServiceBase::pythonStarting() { (void)pythonStartTime(); }
void TimingServiceBase::pythonFinished() { (void)pythonEndTime(); }

void TimingServiceBase::servicesStarting() { (void)servicesStartTime(); }

std::chrono::steady_clock::time_point TimingServiceBase::pythonStartTime() {
  static const std::chrono::steady_clock::time_point s_pythonStartTime = std::chrono::steady_clock::now();
  return s_pythonStartTime;
}

std::chrono::steady_clock::time_point TimingServiceBase::pythonEndTime() {
  static const std::chrono::steady_clock::time_point s_pythonEndTime = std::chrono::steady_clock::now();
  return s_pythonEndTime;
}

std::chrono::steady_clock::time_point TimingServiceBase::servicesStartTime() {
  static const std::chrono::steady_clock::time_point s_servicesStartTime = std::chrono::steady_clock::now();
  return s_servicesStartTime;
}

//
// constructors and destructor
//
TimingServiceBase::TimingServiceBase() = default;

TimingServiceBase::~TimingServiceBase() = default;
