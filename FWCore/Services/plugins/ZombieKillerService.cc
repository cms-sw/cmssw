// -*- C++ -*-
//
// Package:     FWCore/Services
// Class  :     ZombieKillerService
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Sat, 22 Mar 2014 16:25:47 GMT
//

// system include files
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <exception>

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"


namespace edm {
  class ZombieKillerService {
  public:
    ZombieKillerService(edm::ParameterSet const&, edm::ActivityRegistry&);
    
    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    const unsigned int m_checkThreshold;
    const unsigned int m_secsBetweenChecks;
    std::thread m_watchingThread;
    std::condition_variable m_jobDoneCondition;
    std::mutex m_jobDoneMutex;
    bool m_jobDone;
    std::atomic<bool> m_stillAlive;
    std::atomic<unsigned int> m_numberChecksWhenNotAlive;
    
    
    void notAZombieYet();
    void checkForZombie();
    void startThread();
    void stopThread();
  };
}

using namespace edm;

inline
bool isProcessWideService(ZombieKillerService const*) {
  return true;
}


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ZombieKillerService::ZombieKillerService(edm::ParameterSet const& iPSet, edm::ActivityRegistry& iRegistry):
m_checkThreshold(iPSet.getUntrackedParameter<unsigned int>("numberOfAllowedFailedChecksInARow")),
m_secsBetweenChecks(iPSet.getUntrackedParameter<unsigned int>("secondsBetweenChecks")),
m_jobDone(false),
m_stillAlive(true),
m_numberChecksWhenNotAlive(0)
{
  iRegistry.watchPostBeginJob([this](){ startThread(); } );
  iRegistry.watchPostEndJob([this]() {stopThread(); } );
  
  iRegistry.watchPreSourceRun([this](){notAZombieYet();});
  iRegistry.watchPostSourceRun([this](){notAZombieYet();});
  
  iRegistry.watchPreSourceLumi([this](){notAZombieYet();});
  iRegistry.watchPostSourceLumi([this](){notAZombieYet();});

  iRegistry.watchPreSourceEvent([this](StreamID){notAZombieYet();});
  iRegistry.watchPostSourceEvent([this](StreamID){notAZombieYet();});
  
  iRegistry.watchPreModuleBeginStream([this](StreamContext const&, ModuleCallingContext const&){notAZombieYet();});
  iRegistry.watchPostModuleBeginStream([this](StreamContext const&, ModuleCallingContext const&){notAZombieYet();});

  iRegistry.watchPreModuleEndStream([this](StreamContext const&, ModuleCallingContext const&){notAZombieYet();});
  iRegistry.watchPostModuleEndStream([this](StreamContext const&, ModuleCallingContext const&){notAZombieYet();});
  
  iRegistry.watchPreModuleEndJob([this](ModuleDescription const&) {notAZombieYet();});
  iRegistry.watchPostModuleEndJob([this](ModuleDescription const&) {notAZombieYet();});
  iRegistry.watchPreModuleEvent([this](StreamContext const&, ModuleCallingContext const&){notAZombieYet();});
  iRegistry.watchPostModuleEvent([this](StreamContext const&, ModuleCallingContext const&){notAZombieYet();});

  iRegistry.watchPreModuleStreamBeginRun([this](StreamContext const&, ModuleCallingContext const&){notAZombieYet();});
  iRegistry.watchPostModuleStreamBeginRun([this](StreamContext const&, ModuleCallingContext const&){notAZombieYet();});

  iRegistry.watchPreModuleStreamEndRun([this](StreamContext const&, ModuleCallingContext const&){notAZombieYet();});
  iRegistry.watchPostModuleStreamEndRun([this](StreamContext const&, ModuleCallingContext const&){notAZombieYet();});

  iRegistry.watchPreModuleStreamBeginLumi([this](StreamContext const&, ModuleCallingContext const&){notAZombieYet();});
  iRegistry.watchPostModuleStreamBeginLumi([this](StreamContext const&, ModuleCallingContext const&){notAZombieYet();});
  
  iRegistry.watchPreModuleStreamEndLumi([this](StreamContext const&, ModuleCallingContext const&){notAZombieYet();});
  iRegistry.watchPostModuleStreamEndLumi([this](StreamContext const&, ModuleCallingContext const&){notAZombieYet();});

  iRegistry.watchPreModuleGlobalBeginRun([this](GlobalContext const&, ModuleCallingContext const&){notAZombieYet();});
  iRegistry.watchPostModuleGlobalBeginRun([this](GlobalContext const&, ModuleCallingContext const&){notAZombieYet();});
  
  iRegistry.watchPreModuleGlobalEndRun([this](GlobalContext const&, ModuleCallingContext const&){notAZombieYet();});
  iRegistry.watchPostModuleGlobalEndRun([this](GlobalContext const&, ModuleCallingContext const&){notAZombieYet();});
  
  iRegistry.watchPreModuleGlobalBeginLumi([this](GlobalContext const&, ModuleCallingContext const&){notAZombieYet();});
  iRegistry.watchPostModuleGlobalBeginLumi([this](GlobalContext const&, ModuleCallingContext const&){notAZombieYet();});
  
  iRegistry.watchPreModuleGlobalEndLumi([this](GlobalContext const&, ModuleCallingContext const&){notAZombieYet();});
  iRegistry.watchPostModuleGlobalEndLumi([this](GlobalContext const&, ModuleCallingContext const&){notAZombieYet();});

  
}

// ZombieKillerService::ZombieKillerService(const ZombieKillerService& rhs)
// {
//    // do actual copying here;
// }

//ZombieKillerService::~ZombieKillerService()
//{
//}

//
// assignment operators
//
// const ZombieKillerService& ZombieKillerService::operator=(const ZombieKillerService& rhs)
// {
//   //An exception safe implementation is
//   ZombieKillerService temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
ZombieKillerService::notAZombieYet() {
  m_numberChecksWhenNotAlive = 0;
  m_stillAlive = true;
}

void
ZombieKillerService::checkForZombie() {
  if (not m_stillAlive) {
    ++m_numberChecksWhenNotAlive;
    if(m_numberChecksWhenNotAlive > m_checkThreshold) {
      edm::LogError("JobStuck")<<"Too long since the job has last made progress.";
      std::terminate();
    } else {
      edm::LogWarning("JobProgressing")<<"It has been "<<m_numberChecksWhenNotAlive*m_secsBetweenChecks<<" seconds since job seen progressing";
    }
  }
  m_stillAlive = false;
}

void
ZombieKillerService::startThread() {
  m_watchingThread = std::thread([this]() {

    std::unique_lock<std::mutex> lock(m_jobDoneMutex);
    while(not m_jobDoneCondition.wait_for(lock,
                                          std::chrono::seconds(m_secsBetweenChecks),
                                          [this]()->bool
                                          {
                                            return m_jobDone;
                                          }))
    {
      //we timed out
      checkForZombie();
    }
  });
}

void
ZombieKillerService::stopThread() {
  {
    std::lock_guard<std::mutex> guard(m_jobDoneMutex);
    m_jobDone=true;
  }
  m_jobDoneCondition.notify_all();
  m_watchingThread.join();
}

void
ZombieKillerService::fillDescriptions(ConfigurationDescriptions& descriptions) {
  ParameterSetDescription desc;
  desc.addUntracked<unsigned int>("secondsBetweenChecks", 60)->setComment("Number of seconds to wait between checking if progress has been made.");
  desc.addUntracked<unsigned int>("numberOfAllowedFailedChecksInARow", 3)->setComment("Number of allowed checks in a row with no progress.");
  descriptions.add("ZombieKillerService", desc);
}

//
// const member functions
//

//
// static member functions
//

DEFINE_FWK_SERVICE(ZombieKillerService);