// -*- C++ -*-
//
// Package: FWCore/Services
// Class  : StallMonitor
//
// Implementation:
//
// Original Author:  Kyle Knoepfel
//

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "tbb/concurrent_queue.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <sstream>

namespace {
  using clock_t = std::chrono::steady_clock;
  auto now = clock_t::now;

  class FileHandle {
  public:
    FileHandle(std::string const& name) : file_{name} {}
    ~FileHandle(){ file_.close(); }

    template <typename T>
    decltype(auto) operator<<(T const& msg) { return file_ << msg; }
  private:
    std::ofstream file_;
  };
}

namespace edm {

  namespace service {
    class StallMonitor {
    public:
      StallMonitor(ParameterSet const&, ActivityRegistry&);
      ~StallMonitor();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    private:
      void postModuleEventPrefetching(StreamContext const&, ModuleCallingContext const&);
      void preModuleEvent(StreamContext const&, ModuleCallingContext const&);
      void postModuleEvent(StreamContext const&, ModuleCallingContext const&);

      void enqueueDuration(unsigned, unsigned, unsigned);
      std::string assembleMessage(unsigned, unsigned, unsigned);
      void logDuration(std::string);

      FileHandle file_;
      std::vector<decltype(now())> start_ {};
      tbb::concurrent_queue<std::string> loggedDurations_ {};
      std::atomic<bool> durationBeingLogged_ {false};
    };
  }
}

namespace edm {
  namespace service {

    StallMonitor::StallMonitor(ParameterSet const& iPS, ActivityRegistry& iRegistry)
      : file_{iPS.getUntrackedParameter<std::string>("filename")}
    {
      iRegistry.watchPostModuleEventPrefetching(this, &StallMonitor::postModuleEventPrefetching);
      iRegistry.watchPreModuleEvent(this, &StallMonitor::preModuleEvent);
      iRegistry.watchPostModuleEvent(this, &StallMonitor::postModuleEvent);

      iRegistry.preallocateSignal_.connect([this](service::SystemBounds const& iBounds){
          start_.resize(iBounds.maxNumberOfStreams());
        });
    }

    void StallMonitor::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.addUntracked<std::string>("filename", "stallMonitor.txt");
      descriptions.add("StallMonitor", desc);
      descriptions.setComment("This service reports keeps track of various times in event-processing to determine which modules are stalling.");
    }

    void StallMonitor::postModuleEventPrefetching(StreamContext const& sc, ModuleCallingContext const&)
    {
      auto const i = sc.streamID().value();
      start_[i] = now();
    }

    void StallMonitor::preModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc)
    {
      auto const i = sc.streamID().value();
      auto const prefetch_to_eventStart = (now()-start_[i]).count();
      enqueueDuration(1, mcc.moduleDescription()->id(), prefetch_to_eventStart);
    }

    void StallMonitor::postModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc)
    {
      auto const i = sc.streamID().value();
      auto const event_start_to_end = (now()-start_[i]).count();
      enqueueDuration(2, mcc.moduleDescription()->id(), event_start_to_end);
    }

    std::string StallMonitor::assembleMessage(unsigned const mode, unsigned const moduleID, unsigned const time)
    {
      std::ostringstream oss;
      oss << mode << ' ' << moduleID << ' ' << time;
      return oss.str();
    }

    StallMonitor::~StallMonitor()
    {
      std::string msg;
      while (loggedDurations_.try_pop(msg)) {
        file_ << msg << '\n';
      }
    }

    void StallMonitor::logDuration(std::string msg)
    {
      bool expected {false};
      if(durationBeingLogged_.compare_exchange_strong(expected,true)) {
        do {
          file_ << msg << '\n';
        } while (loggedDurations_.try_pop(msg));
        durationBeingLogged_.store(false);
      } else {
        loggedDurations_.push(std::move(msg));
      }
    }

    void StallMonitor::enqueueDuration(unsigned const mode, unsigned const moduleID, unsigned const time)
    {
      auto const& msg = assembleMessage(mode, moduleID, time);
      logDuration(msg);
    }

  }
}

using edm::service::StallMonitor;
DEFINE_FWK_SERVICE(StallMonitor);
