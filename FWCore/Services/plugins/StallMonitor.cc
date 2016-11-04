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

#include "tbb/concurrent_unordered_map.h"

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
  template <typename T>
  std::enable_if_t<std::is_integral<T>::value>
  concatenate(std::ostream& os, T const t)
  {
    os << ' ' << t;
  }

  template <typename H, typename... T>
  std::enable_if_t<std::is_integral<H>::value>
  concatenate(std::ostream& os, H const h, T const... t)
  {
    os << ' ' << h;
    concatenate(os, t...);
  }

  enum class step : char { preEventReadFromSource = 'A',
                           postEventReadFromSource,
                           preEvent,
                           postEvent,
                           postModuleEventPrefetching,
                           preModuleEvent,
                           postModuleEvent };

  template <step S, typename... ARGS>
  std::string assembleMessage(ARGS const... args)
  {
    std::ostringstream oss;
    oss << static_cast<std::underlying_type_t<step>>(S);
    concatenate(oss, args...);
    oss << '\n';
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
      bool validFile_; // Separate data member from file to improve efficiency
      std::chrono::milliseconds const stallThreshold_;
      decltype(now()) beginTime_ {};

      using StreamID_value = decltype(std::declval<StreamID>().value());
      using ModuleID = decltype(std::declval<ModuleDescription>().id());
      tbb::concurrent_unordered_map<std::pair<StreamID_value,ModuleID>, decltype(beginTime_)> stallStart_ {};

      std::vector<std::string> moduleLabels_ {}; // Temporary, used to seed moduleStats_.
      std::vector<StallStatistics> moduleStats_ {};
    };

  }

}

namespace {
  constexpr char const* filename_default {""};
  constexpr double threshold_default {0.1};
}

using edm::service::StallMonitor;
using namespace std::chrono;

StallMonitor::StallMonitor(ParameterSet const& iPS, ActivityRegistry& iRegistry)
  : file_{iPS.getUntrackedParameter<std::string>("filename", filename_default)}
  , validFile_{file_}
  , stallThreshold_{static_cast<long int>(iPS.getUntrackedParameter<double>("stallThreshold", threshold_default)*1000)}
{
  iRegistry.watchPreModuleConstruction(this, &StallMonitor::preModuleConstruction);
  iRegistry.watchPostBeginJob(this, &StallMonitor::postBeginJob);
  iRegistry.watchPostModuleEventPrefetching(this, &StallMonitor::postModuleEventPrefetching);
  iRegistry.watchPreEvent(this, &StallMonitor::preEvent);
  iRegistry.watchPreModuleEvent(this, &StallMonitor::preModuleEvent);
  iRegistry.watchPreEventReadFromSource(this, &StallMonitor::preEventReadFromSource);
  iRegistry.watchPostEventReadFromSource(this, &StallMonitor::postEventReadFromSource);
  iRegistry.watchPostModuleEvent(this, &StallMonitor::postModuleEvent);
  iRegistry.watchPostEvent(this, &StallMonitor::postEvent);
  iRegistry.watchPostEndJob(this, &StallMonitor::postEndJob);

  if (validFile_) {
    // PRINT TABLE AT TOP LISTING ENUMERATIONS
  }

}

void StallMonitor::fillDescriptions(ConfigurationDescriptions& descriptions)
{
  ParameterSetDescription desc;
  desc.addUntracked<std::string>("fileName", filename_default)->setComment("Name of file to which detailed timing information should be written.\n"
                                                                           "An empty filename argument (the default) indicates that no extra\n"
                                                                           "information will be written to a dedicated file, but only the summary\n"
                                                                           "including stalling-modules information will be logged.");
  desc.addUntracked<double>("stallThreshold", threshold_default)->setComment("Threshold (in seconds) used to classify modules as stalled.\n"
                                                                             "Millisecond granularity allowed.");
  descriptions.add("StallMonitor", desc);
  descriptions.setComment("This service keeps track of various times in event-processing to determine which modules are stalling.");
}

void StallMonitor::preModuleConstruction(ModuleDescription const& md)
{
  // Module labels are dense, so if the module id is greater than the
  // size of moduleLabels_, grow the vector to the right index, and
  // assign the last entry to the desired label.
  auto const mid = md.id();
  if (mid < moduleLabels_.size()) {
    moduleLabels_[mid] = md.moduleLabel();
  }
  else {
    moduleLabels_.resize(mid+1);
    moduleLabels_.back() = md.moduleLabel();
  }
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

void StallMonitor::preEvent(StreamContext const& sc)
{
  if (!validFile_) return;
  auto const t = duration_cast<milliseconds>(now()-beginTime_).count();
  auto const& eid = sc.eventID();
  auto msg = assembleMessage<step::preEvent>(stream_id(sc), eid.run(), eid.luminosityBlock(), eid.event(), t);
  file_.write(std::move(msg));
}

void StallMonitor::postModuleEventPrefetching(StreamContext const& sc, ModuleCallingContext const& mcc)
{
  auto const sid = stream_id(sc);
  auto const mid = module_id(mcc);
  auto start = stallStart_[std::make_pair(sid,mid)] = now();

  if (!validFile_) return;
  auto const t = duration_cast<milliseconds>(start-beginTime_).count();
  auto msg = assembleMessage<step::postModuleEventPrefetching>(sid, mid, t);
  file_.write(std::move(msg));
}

void StallMonitor::preModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc)
{
  auto const preModEvent = now();
  auto const sid = stream_id(sc);
  auto const mid = module_id(mcc);
  if (file_) {
    auto msg = assembleMessage<step::preModuleEvent>(sid, mid, duration_cast<milliseconds>(preModEvent-beginTime_).count());
    file_.write(std::move(msg));
  }

  auto const preFetch_to_preModEvent = duration_cast<milliseconds>(preModEvent-stallStart_[std::make_pair(sid,mid)]);
  if (preFetch_to_preModEvent < stallThreshold_) return;
  moduleStats_[mid].update(preFetch_to_preModEvent);
}

void StallMonitor::preEventReadFromSource(StreamContext const& sc, ModuleCallingContext const& mcc)
{
  if (!validFile_) return;
  auto const t = duration_cast<milliseconds>(now()-beginTime_).count();
  auto msg = assembleMessage<step::preEventReadFromSource>(stream_id(sc), module_id(mcc), t);
  file_.write(std::move(msg));
}

void StallMonitor::postEventReadFromSource(StreamContext const& sc, ModuleCallingContext const& mcc)
{
  if (!validFile_) return;
  auto const t = duration_cast<milliseconds>(now()-beginTime_).count();
  auto msg = assembleMessage<step::postEventReadFromSource>(stream_id(sc), module_id(mcc), t);
  file_.write(std::move(msg));
}

void StallMonitor::postModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc)
{
  if (!validFile_) return;
  auto const postModEvent = duration_cast<milliseconds>(now()-beginTime_).count();
  auto msg = assembleMessage<step::postModuleEvent>(stream_id(sc), module_id(mcc), postModEvent);
  file_.write(std::move(msg));
}

void StallMonitor::postEvent(StreamContext const& sc)
{
  if (!validFile_) return;
  auto const t = duration_cast<milliseconds>(now()-beginTime_).count();
  auto const& eid = sc.eventID();
  auto msg = assembleMessage<step::postEvent>(stream_id(sc), eid.run(), eid.luminosityBlock(), eid.event(), t);
  file_.write(std::move(msg));
}

void StallMonitor::postEndJob()
{
  // Prepare summary
  std::size_t width {};
  edm::for_all(moduleStats_, [&width](auto const& stats) { width = std::max(width, stats.label().size()); });

  Column tag {"StallMonitor>"};
  Column col1 {"Module label", width};
  Column col2 {"# of stalls"};
  Column col3 {"Total stalled time"};
  Column col4 {"Max stalled time"};

  LogAbsolute out {"StallMonitor"};
  out << tag << col1 << col2 << col3 << col4 << '\n';
  //  out << std::string('-',width+col1+col2+col3) << '\n';
  for (auto const& stats : moduleStats_) {
    out << tag
        << col1(stats.label())
        << col2(stats.numberOfStalls())
        << col3(stats.totalStalledTime())
        << col4(stats.maxStalledTime()) << '\n';
  }

  if (validFile_) {
    // FIXME: DUMP MODULE NAMES/IDS
  }
}

DEFINE_FWK_SERVICE(StallMonitor);
