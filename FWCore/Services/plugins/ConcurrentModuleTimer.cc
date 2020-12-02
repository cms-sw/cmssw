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
#include <memory>

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
#include "DataFormats/Provenance/interface/ModuleDescription.h"

namespace edm {
  namespace service {
    class ConcurrentModuleTimer {
    public:
      ConcurrentModuleTimer(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iAR);
      ~ConcurrentModuleTimer();
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    private:
      void start();
      void stop();

      bool trackModule(ModuleCallingContext const& iContext) const;
      std::unique_ptr<std::atomic<std::chrono::high_resolution_clock::rep>[]> m_timeSums;
      std::vector<std::string> m_modulesToExclude;
      std::vector<unsigned int> m_excludedModuleIds;
      std::chrono::high_resolution_clock::time_point m_time;
      unsigned int m_nTimeSums = 0;
      unsigned int m_nModules;
      unsigned int m_maxNModules = 0;
      const unsigned int m_padding;
      std::atomic<bool> m_spinLock;
      bool m_startedTiming;
      const bool m_excludeSource;
      const bool m_trackGlobalBeginRun;
    };
  }  // namespace service
}  // namespace edm

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
ConcurrentModuleTimer::ConcurrentModuleTimer(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iReg)
    : m_modulesToExclude(iConfig.getUntrackedParameter<std::vector<std::string>>("modulesToExclude")),
      m_time(),
      m_nModules(0),
      m_padding(iConfig.getUntrackedParameter<unsigned int>("padding")),
      m_spinLock{false},
      m_startedTiming(false),
      m_excludeSource(iConfig.getUntrackedParameter<bool>("excludeSource")),
      m_trackGlobalBeginRun(iConfig.getUntrackedParameter<bool>("trackGlobalBeginRun")) {
  if (not m_modulesToExclude.empty()) {
    iReg.watchPreModuleConstruction([this](ModuleDescription const& iMod) {
      for (auto const& name : m_modulesToExclude) {
        if (iMod.moduleLabel() == name) {
          m_excludedModuleIds.push_back(iMod.id());
          break;
        }
      }
    });
    iReg.watchPreModuleDestruction([this](ModuleDescription const& iMod) {
      auto found = std::find(m_excludedModuleIds.begin(), m_excludedModuleIds.end(), iMod.id());
      if (found != m_excludedModuleIds.end()) {
        m_excludedModuleIds.erase(found);
      }
    });
    iReg.watchPreModuleEvent([this](StreamContext const&, ModuleCallingContext const& iContext) {
      if (trackModule(iContext)) {
        start();
      }
    });
    iReg.watchPostModuleEvent([this](StreamContext const&, ModuleCallingContext const& iContext) {
      if (trackModule(iContext)) {
        stop();
      }
    });

    iReg.watchPreModuleEventDelayedGet([this](StreamContext const&, ModuleCallingContext const& iContext) {
      if (trackModule(iContext)) {
        if (iContext.state() == ModuleCallingContext::State::kRunning) {
          stop();
        }
      }
    });
    iReg.watchPostModuleEventDelayedGet([this](StreamContext const&, ModuleCallingContext const& iContext) {
      if (trackModule(iContext)) {
        if (iContext.state() == ModuleCallingContext::State::kRunning) {
          start();
        }
      }
    });

  } else {
    //apply to all modules so can use faster version
    iReg.watchPreModuleEvent([this](StreamContext const&, ModuleCallingContext const&) { start(); });
    iReg.watchPostModuleEvent([this](StreamContext const&, ModuleCallingContext const&) { stop(); });

    iReg.watchPreModuleEventDelayedGet([this](StreamContext const&, ModuleCallingContext const& iContext) {
      if (iContext.state() == ModuleCallingContext::State::kRunning) {
        stop();
      }
    });
    iReg.watchPostModuleEventDelayedGet([this](StreamContext const&, ModuleCallingContext const& iContext) {
      if (iContext.state() == ModuleCallingContext::State::kRunning) {
        start();
      }
    });
    if (m_trackGlobalBeginRun) {
      iReg.watchPreModuleGlobalBeginRun([this](GlobalContext const&, ModuleCallingContext const&) {
        if (not m_startedTiming) {
          m_time = std::chrono::high_resolution_clock::now();
          m_startedTiming = true;
        }

        start();
      });
      iReg.watchPostModuleGlobalBeginRun([this](GlobalContext const&, ModuleCallingContext const&) { stop(); });
    }
  }

  iReg.watchPreallocate([this](edm::service::SystemBounds const& iBounds) {
    m_nTimeSums = iBounds.maxNumberOfThreads() + 1 + m_padding;
    m_timeSums = std::make_unique<std::atomic<std::chrono::high_resolution_clock::rep>[]>(m_nTimeSums);
    for (unsigned int i = 0; i < m_nTimeSums; ++i) {
      m_timeSums[i] = 0;
    }
  });

  iReg.watchPreSourceEvent([this](StreamID) {
    if (not m_startedTiming) {
      m_time = std::chrono::high_resolution_clock::now();
      m_startedTiming = true;
    }
    if (not m_excludeSource) {
      start();
    }
  });
  if (not m_excludeSource) {
    iReg.watchPostSourceEvent([this](StreamID) { stop(); });
  }
}

ConcurrentModuleTimer::~ConcurrentModuleTimer() {
  std::cout << "Maximum concurrent running modules: " << m_maxNModules << std::endl;
  std::cout << "Fraction of time running n Modules simultaneously" << std::endl;
  for (unsigned int i = 0; i < m_nTimeSums; ++i) {
    std::cout << i << " " << m_timeSums[i] / double(m_timeSums[0]) << " " << m_timeSums[i] << std::endl;
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
void ConcurrentModuleTimer::start() {
  auto const newTime = std::chrono::high_resolution_clock::now();
  std::chrono::high_resolution_clock::time_point oldTime;
  bool expected = false;
  unsigned int nModules;
  while (not m_spinLock.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
    expected = false;
  }
  {
    oldTime = m_time;
    m_time = newTime;
    nModules = ++m_nModules;
    if (nModules > m_maxNModules) {
      m_maxNModules = nModules;
    }
    m_spinLock.store(false, std::memory_order_release);
  }
  assert(nModules < m_nTimeSums);
  auto diff = newTime - oldTime;
  for (unsigned int i = 0; i < nModules; ++i) {
    m_timeSums[i].fetch_add(diff.count());
  }
}

void ConcurrentModuleTimer::stop() {
  auto const newTime = std::chrono::high_resolution_clock::now();
  std::chrono::high_resolution_clock::time_point oldTime;
  bool expected = false;
  unsigned int nModules;
  while (not m_spinLock.compare_exchange_weak(expected, true, std::memory_order_acq_rel)) {
    expected = false;
  }
  {
    oldTime = m_time;
    m_time = newTime;
    nModules = m_nModules--;
    m_spinLock.store(false, std::memory_order_release);
  }
  assert(nModules < m_nTimeSums);
  auto diff = newTime - oldTime;
  for (unsigned int i = 0; i <= nModules; ++i) {
    m_timeSums[i].fetch_add(diff.count());
  }
}

//
// const member functions
//
bool ConcurrentModuleTimer::trackModule(ModuleCallingContext const& iContext) const {
  auto modId = iContext.moduleDescription()->id();
  for (auto const id : m_excludedModuleIds) {
    if (modId == id) {
      return false;
    }
  }
  return true;
}

//
// static member functions
//
void ConcurrentModuleTimer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<std::string>>("modulesToExclude", std::vector<std::string>{})
      ->setComment("Module labels to exclude from the timing measurements");
  desc.addUntracked<bool>("excludeSource", false)->setComment("Exclude the time the source is running");
  desc.addUntracked<unsigned int>("padding", 0)
      ->setComment(
          "[Expert use only] Extra possible concurrent modules beyond thread count.\n Only useful in debugging "
          "possible framework scheduling problems.");
  desc.addUntracked<bool>("trackGlobalBeginRun", false)
      ->setComment("Check for concurrent modules during global begin run");
  descriptions.add("ConcurrentModuleTimer", desc);
}

DEFINE_FWK_SERVICE(ConcurrentModuleTimer);
