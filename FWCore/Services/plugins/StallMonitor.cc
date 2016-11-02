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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/Column.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

namespace {

  using clock_t = std::chrono::steady_clock;
  auto now = clock_t::now;

  inline auto stream_id(edm::StreamContext const& cs)
  {
    return cs.streamID().value();
  }

  inline auto module_id(edm::ModuleCallingContext const& mcc)
  {
    return mcc.moduleDescription()->id();
  }

  //===============================================================
  class StallStatistics {
  public:

    // c'tor receiving 'std::string const&' type not provided since we
    // must be able to call (e.g.) std::vector<StallStatistics>(20),
    // for which a default label is not sensible in this context.

    std::string const& label() const { return label_; }
    unsigned numberOfStalls() const { return stallCounter_; }

    using rep_t = std::chrono::milliseconds::rep;
    rep_t totalStalledTime() const { return totalTime_; }
    rep_t maxStalledTime() const { return maxTime_; }

    // Modifiers
    void setLabel(std::string const& label) { label_ = label; }

    void update(std::chrono::milliseconds const ms)
    {
      ++stallCounter_;
      auto const thisTime = ms.count();
      totalTime_ += thisTime;
      rep_t max {maxTime_};
      while (thisTime > max && !maxTime_.compare_exchange_weak(max, thisTime)) {}
    }

  private:
    std::string label_ {};
    std::atomic<unsigned> stallCounter_ {};
    std::atomic<rep_t> totalTime_ {};
    std::atomic<rep_t> maxTime_ {};
  };

  //===============================================================
  // Message-assembly utilities
  struct Quantity {
    template <typename T>
    Quantity(std::string const& field, T const& t)
    {
      std::ostringstream msg;
      msg << field << ": " << t;
      data = msg.str();
    }
    std::string data;
  };

  std::ostream& operator<<(std::ostream& os, Quantity const& q)
  {
    return os << q.data;
  }

  template <typename T>
  void concatenate(std::ostream& os, T const& t)
  {
    os << ' ' << t;
  }

  template <typename H, typename... T>
  void concatenate(std::ostream& os, H const& h, T const&... t)
  {
    os << ' ' << h;
    concatenate(os, t...);
  }

  template <typename... ARGS>
  std::string assembleMessage(std::string const& step, ARGS const&... args)
  {
    std::ostringstream oss;
    concatenate(oss, args...);
    return step + ":" + oss.str() + "\n";
  }

}

namespace edm {
  namespace service {

    class StallMonitor {
    public:
      StallMonitor(ParameterSet const&, ActivityRegistry&);
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    private:
      void preModuleConstruction(edm::ModuleDescription const&);
      void preEvent(StreamContext const&);
      void postEvent(StreamContext const&);
      void postModuleEventPrefetching(StreamContext const&, ModuleCallingContext const&);
      void preEventReadFromSource(StreamContext const&, ModuleCallingContext const&);
      void postEventReadFromSource(StreamContext const&, ModuleCallingContext const&);
      void preModuleEvent(StreamContext const&, ModuleCallingContext const&);
      void postModuleEvent(StreamContext const&, ModuleCallingContext const&);
      void postBeginJob();
      void postEndJob();

      ThreadSafeOutputFileStream file_;
      std::chrono::milliseconds const stallThreshold_;
      decltype(now()) beginTime_ {};
      std::vector<decltype(beginTime_)> stallStart_ {}; // One stall-start time per stream
      std::vector<std::string> moduleLabels_ {}; // Temporary, used to seed moduleStats_.
      std::vector<StallStatistics> moduleStats_ {};
    };

  }
}

using edm::service::StallMonitor;
using namespace std::chrono;

StallMonitor::StallMonitor(ParameterSet const& iPS, ActivityRegistry& iRegistry)
  : file_{iPS.getUntrackedParameter<std::string>("filename")}
  , stallThreshold_{static_cast<long int>(iPS.getUntrackedParameter<double>("stallThreshold", 0.1)*1000)}
{
  iRegistry.watchPreModuleConstruction(this, &StallMonitor::preModuleConstruction);
  iRegistry.watchPostModuleEventPrefetching(this, &StallMonitor::postModuleEventPrefetching);
  iRegistry.watchPreEvent(this, &StallMonitor::preEvent);
  iRegistry.watchPostEvent(this, &StallMonitor::postEvent);
  iRegistry.watchPreModuleEvent(this, &StallMonitor::preModuleEvent);
  iRegistry.watchPostModuleEvent(this, &StallMonitor::postModuleEvent);
  iRegistry.watchPreEventReadFromSource(this, &StallMonitor::preEventReadFromSource);
  iRegistry.watchPostEventReadFromSource(this, &StallMonitor::postEventReadFromSource);

  iRegistry.preallocateSignal_.connect([this](auto const& iBounds){ stallStart_.resize(iBounds.maxNumberOfStreams());});
  iRegistry.watchPostBeginJob(this, &StallMonitor::postBeginJob);
  iRegistry.watchPostEndJob(this, &StallMonitor::postEndJob);
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

void StallMonitor::preModuleConstruction(ModuleDescription const& md)
{
  moduleLabels_.push_back(md.moduleLabel());
  assert(md.id()+1 == moduleStats_.size());
}

void StallMonitor::postBeginJob()
{
  // Since a (push,emplace)_back cannot be called for a vector of a
  // type containing atomics (like 'StallStatistics')--i.e. atomics
  // have no copy/move-assignment operators, we must specify the size
  // of the vector at construction time.
  moduleStats_ = std::vector<StallStatistics>(moduleLabels_.size());
  for (std::size_t i{}; i < moduleStats_.size(); ++i) {
    moduleStats_[i].setLabel(moduleLabels_[i]);
  }
  moduleLabels_.clear();
  beginTime_ = now();
}

void StallMonitor::preEventReadFromSource(StreamContext const& sc, ModuleCallingContext const& mcc)
{
  auto const t = (now()-beginTime_).count();
  file_ << assembleMessage("preEventReadFromSource",
                           Quantity{"Stream id", stream_id(sc)},
                           Quantity{"Module id", module_id(mcc)},
                           Quantity{"Time", t});
}

void StallMonitor::postEventReadFromSource(StreamContext const& sc, ModuleCallingContext const& mcc)
{
  auto const t = (now()-beginTime_).count();
  file_ << assembleMessage("postEventReadFromSource",
                           Quantity{"Stream id", stream_id(sc)},
                           Quantity{"Module id", module_id(mcc)},
                           Quantity{"Time", t});
}

void StallMonitor::preEvent(StreamContext const& sc)
{
  auto const t = (now()-beginTime_).count();
  file_ << assembleMessage("preEvent",
                           Quantity{"Stream id", stream_id(sc)},
                           sc.eventID(),
                           Quantity{"Time", t});
}

void StallMonitor::postEvent(StreamContext const& sc)
{
  auto const t = (now()-beginTime_).count();
  file_ << assembleMessage("postEvent",
                           Quantity{"Stream id", stream_id(sc)},
                           sc.eventID(),
                           Quantity{"Time", t});
}

void StallMonitor::postModuleEventPrefetching(StreamContext const& sc, ModuleCallingContext const& mcc)
{
  auto const i = stream_id(sc);
  stallStart_[i] = now();
  file_ << assembleMessage("postModuleEventPrefetching",
                           Quantity{"Stream id", i},
                           Quantity{"Module id", module_id(mcc)},
                           Quantity{"Time", (stallStart_[i]-beginTime_).count()});
}

void StallMonitor::preModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc)
{
  auto const preModEvent = now();
  auto const i = stream_id(sc);
  auto const mid = module_id(mcc);
  file_ << assembleMessage("preModuleEvent",
                           Quantity{"Stream id", i},
                           Quantity{"Module id", mid},
                           Quantity{"Time", (preModEvent-beginTime_).count()});

  auto const preFetch_to_preModEvent = duration_cast<milliseconds>(preModEvent-stallStart_[i]);
  if (preFetch_to_preModEvent < stallThreshold_) return;
  moduleStats_[mid].update(preFetch_to_preModEvent);
}

void StallMonitor::postModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc)
{
  auto const postModEvent = (now()-beginTime_).count();
  file_ << assembleMessage("postModuleEvent",
                           Quantity{"Stream id", stream_id(sc)},
                           Quantity{"Module id", module_id(mcc)},
                           Quantity{"Time", postModEvent});
}

void StallMonitor::postEndJob()
{
  // Prepare summary
  std::size_t width {};
  edm::for_all(moduleStats_, [&width](auto const& stats) { width = std::max(width, stats.label().size()); });

  Column col1 {"Module label", width};
  Column col2 {"# of stalls"};
  Column col3 {"Total stalled time"};
  Column col4 {"Max stalled time"};

  LogAbsolute out {"StallMonitor"};
  out << col1 << col2 << col3 << col4 << '\n';
  //  out << std::string('-',width+col1+col2+col3) << '\n';
  for (auto const& stats : moduleStats_) {
    out << col1(stats.label())
        << col2(stats.numberOfStalls())
        << col3(stats.totalStalledTime())
        << col4(stats.maxStalledTime()) << '\n';
  }
}

DEFINE_FWK_SERVICE(StallMonitor);
