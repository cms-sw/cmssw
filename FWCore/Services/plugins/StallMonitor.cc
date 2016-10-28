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
#include "FWCore/Concurrency/interface/ThreadSafeOutputFileStream.h"
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

  std::string assembleMessage(std::string const& mode, unsigned const streamID, unsigned const moduleID, double const time)
  {
    std::ostringstream oss;
    oss << mode << ": Stream " << streamID << " Module " << moduleID << " Time " << time << "s";
    return oss.str();
  }
}

namespace edm {
  namespace service {

    class StallMonitor {
    public:
      StallMonitor(ParameterSet const&, ActivityRegistry&);
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    private:
      void preEvent(StreamContext const&);
      void postModuleEventPrefetching(StreamContext const&, ModuleCallingContext const&);
      void preModuleEvent(StreamContext const&, ModuleCallingContext const&);
      void postModuleEvent(StreamContext const&, ModuleCallingContext const&);

      ThreadSafeOutputFileStream file_;
      std::chrono::milliseconds const stallThreshold_;
      decltype(now()) beginTime_;
      std::vector<decltype(now())> stallStart_ {};
      std::vector<char> shouldLog_ {};
    };

  }
}

using edm::service::StallMonitor;
using namespace std::chrono;

StallMonitor::StallMonitor(ParameterSet const& iPS, ActivityRegistry& iRegistry)
  : file_{iPS.getUntrackedParameter<std::string>("filename")}
  , stallThreshold_{static_cast<long int>(iPS.getUntrackedParameter<double>("stallThreshold", 0.1)*1000)}
{
  iRegistry.watchPostModuleEventPrefetching(this, &StallMonitor::postModuleEventPrefetching);
  iRegistry.watchPreEvent(this, &StallMonitor::preEvent);
  iRegistry.watchPreModuleEvent(this, &StallMonitor::preModuleEvent);
  iRegistry.watchPostModuleEvent(this, &StallMonitor::postModuleEvent);

  auto setPerStreamVectors = [this](service::SystemBounds const& iBounds){
    auto const nStreams = iBounds.maxNumberOfStreams();
    stallStart_.resize(nStreams);
    shouldLog_.resize(nStreams);
  };

  iRegistry.preallocateSignal_.connect(setPerStreamVectors);
  iRegistry.postBeginJobSignal_.connect([this]{ beginTime_ = now(); });
}

void StallMonitor::fillDescriptions(ConfigurationDescriptions& descriptions)
{
  ParameterSetDescription desc;
  desc.addUntracked<std::string>("fileName", "stallMonitor.txt");
  desc.addUntracked<double>("stallThreshold", 0.1)->setComment("Threshold (in seconds) used to classify modules as stalled.\n"
                                                               "Millisecond granularity allowed.");
  descriptions.add("StallMonitor", desc);
  descriptions.setComment("This service keeps track of various times in event-processing to determine which modules are stalling.");
}

void StallMonitor::preEvent(StreamContext const& sc)
{
  auto const i = sc.streamID().value();
  std::ostringstream msg;
  msg << "preEvent: Stream " << i << ' ' << sc.eventID() << '\n';
  file_ << msg.str();
}

void StallMonitor::postModuleEventPrefetching(StreamContext const& sc, ModuleCallingContext const&)
{
  auto const i = sc.streamID().value();
  stallStart_[i] = now();
  shouldLog_[i] = 0;
}

void StallMonitor::preModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc)
{
  auto const i = sc.streamID().value();
  auto const preModEvent_tp = now();
  auto const preFetch_to_preModEvent = duration_cast<milliseconds>(preModEvent_tp-stallStart_[i]);
  if (preFetch_to_preModEvent < stallThreshold_) return;

  shouldLog_[i] = 1;
  auto const mid = mcc.moduleDescription()->id();

  // Would be nice to put the postModuleEventPrefetching message in
  // the corresponding function call.  But since we don't want to log
  // the message unless we make it this far, we log the message here.
  file_ << assembleMessage("postModuleEventPrefetching", i, mid, (stallStart_[i]-beginTime_).count());
  file_ << assembleMessage("preModuleEvent", i, mid, (preModEvent_tp-beginTime_).count());
}

void StallMonitor::postModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc)
{
  auto const i = sc.streamID().value();
  if (!shouldLog_[i]) return;

  auto const postModEvent = (now()-beginTime_).count();
  file_ << assembleMessage("postModuleEvent", i, mcc.moduleDescription()->id(), postModEvent);
}

DEFINE_FWK_SERVICE(StallMonitor);
