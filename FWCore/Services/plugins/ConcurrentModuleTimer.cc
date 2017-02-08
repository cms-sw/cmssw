// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     ConcurrentModuleTimer
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Tue, 10 Dec 2013 21:16:00 GMT
//
#include <vector>
#include <atomic>
#include <chrono>
#include <iostream>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/CPUTimer.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"


namespace edm {
  namespace service {
    class ConcurrentModuleTimer {
    public:
      ConcurrentModuleTimer(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iAR);
      ~ConcurrentModuleTimer();
      //static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    private:
      void start();
      void stop();
      
      std::unique_ptr<std::atomic<std::chrono::high_resolution_clock::rep>[]> m_timeSums;
      std::chrono::high_resolution_clock::time_point m_time;
      unsigned int m_nTimeSums;
      unsigned int m_nModules;
      std::atomic<bool> m_spinLock;
      bool m_startedTiming;
    };
  }
}

using namespace edm::service;
// system include files

// user include files

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ConcurrentModuleTimer::ConcurrentModuleTimer(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iReg):
m_time(),
m_nModules(0),
m_spinLock{false},
m_startedTiming(false)
{
  iReg.watchPreModuleEvent([this](StreamContext const&, ModuleCallingContext const&){
    start();
  });
  iReg.watchPostModuleEvent([this](StreamContext const&, ModuleCallingContext const&){
    stop();
  });
  
  iReg.watchPreModuleEventDelayedGet([this](StreamContext const&, ModuleCallingContext const& iContext){
      if(iContext.state() == ModuleCallingContext::State::kRunning) {
	stop();
      }
  });
  iReg.watchPostModuleEventDelayedGet([this](StreamContext const&, ModuleCallingContext const& iContext){
      if(iContext.state() == ModuleCallingContext::State::kRunning) {
	start();
      }
  });
  
  iReg.watchPreallocate([this](edm::service::SystemBounds const& iBounds){
    m_nTimeSums =iBounds.maxNumberOfThreads()+1;
    m_timeSums.reset(new std::atomic<std::chrono::high_resolution_clock::rep>[m_nTimeSums]);
    for(unsigned int i=0; i<m_nTimeSums;++i) {
      m_timeSums[i]=0;
    }
  });
  
  iReg.watchPreSourceEvent([this](StreamID){
    if(not m_startedTiming) {
      m_time = std::chrono::high_resolution_clock::now();
      m_startedTiming=true;
    }
    start();
  });
  iReg.watchPostSourceEvent([this](StreamID){
    stop();
  });
}

ConcurrentModuleTimer::~ConcurrentModuleTimer() {
  
  std::cout <<"Fraction of time running n Modules simultaneously"<<std::endl;
  for (unsigned int i=0; i<m_nTimeSums; ++i) {
    std::cout <<i<<" "<<m_timeSums[i]/double(m_timeSums[0])<<" "<<m_timeSums[i]<<std::endl;
  }
  
}

// ConcurrentModuleTimer::ConcurrentModuleTimer(const ConcurrentModuleTimer& rhs)
// {
//    // do actual copying here;
// }

//
// assignment operators
//
// const ConcurrentModuleTimer& ConcurrentModuleTimer::operator=(const ConcurrentModuleTimer& rhs)
// {
//   //An exception safe implementation is
//   ConcurrentModuleTimer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
ConcurrentModuleTimer::start()
{
  auto const newTime =std::chrono::high_resolution_clock::now();
  std::chrono::high_resolution_clock::time_point oldTime;
  bool expected = false;
  unsigned int nModules;
  while (not m_spinLock.compare_exchange_strong(expected,true,std::memory_order_acq_rel)){
    expected = false;
  }
  {
    oldTime = m_time;
    m_time = newTime;
    nModules = ++m_nModules;
    m_spinLock.store(false,std::memory_order_release);
  }
  assert(nModules <m_nTimeSums);
  auto diff = newTime - oldTime;
  for(unsigned int i=0;i<nModules;++i) {
    m_timeSums[i].fetch_add(diff.count());
  }
}

void
ConcurrentModuleTimer::stop()
{
  auto const newTime =std::chrono::high_resolution_clock::now();
  std::chrono::high_resolution_clock::time_point oldTime;
  bool expected = false;
  unsigned int nModules;
  while (not m_spinLock.compare_exchange_weak(expected,true,std::memory_order_acq_rel)){
    expected = false;
  }
  {
    oldTime = m_time;
    m_time = newTime;
    nModules = m_nModules--;
    m_spinLock.store(false,std::memory_order_release);
  }
  assert(nModules <m_nTimeSums);
  auto diff = newTime - oldTime;
  for(unsigned int i=0;i<=nModules;++i) {
    m_timeSums[i].fetch_add(diff.count());
  }
}


//
// const member functions
//

//
// static member functions
//
DEFINE_FWK_SERVICE(ConcurrentModuleTimer);

