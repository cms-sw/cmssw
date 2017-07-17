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
#include "FWCore/Utilities/interface/OStreamColumn.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "tbb/concurrent_unordered_map.h"

#include <atomic>
#include <chrono>
#include <iomanip>
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
    StallStatistics() = default;

    std::string const& label() const { return label_; }
    unsigned numberOfStalls() const { return stallCounter_; }

    using duration_t = std::chrono::milliseconds;
    using rep_t = duration_t::rep;

    duration_t totalStalledTime() const { return duration_t{totalTime_.load()}; }
    duration_t maxStalledTime() const { return duration_t{maxTime_.load()}; }

    // Modifiers
    void setLabel(std::string const& label) { label_ = label; }

    void update(std::chrono::milliseconds const ms)
    {
      ++stallCounter_;
      auto const thisTime = ms.count();
      totalTime_ += thisTime;
      rep_t max {maxTime_};
      while (thisTime > max && !maxTime_.compare_exchange_strong(max, thisTime));
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

  enum class step : char { preSourceEvent = 'S',
                           postSourceEvent = 's',
                           preEvent = 'E',
                           postModuleEventPrefetching = 'p',
                           preModuleEvent = 'M',
                           preEventReadFromSource = 'R',
                           postEventReadFromSource = 'r',
                           postModuleEvent = 'm' ,
                           postEvent = 'e'};

  std::ostream& operator<<(std::ostream& os, step const s)
  {
    os << static_cast<std::underlying_type_t<step>>(s);
    return os;
  }

  template <step S, typename... ARGS>
  std::string assembleMessage(ARGS const... args)
  {
    std::ostringstream oss;
    oss << S;
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
      void postBeginJob();
      void preSourceEvent(StreamID);
      void postSourceEvent(StreamID);
      void preEvent(StreamContext const&);
      void preModuleEvent(StreamContext const&, ModuleCallingContext const&);
      void postModuleEventPrefetching(StreamContext const&, ModuleCallingContext const&);
      void preEventReadFromSource(StreamContext const&, ModuleCallingContext const&);
      void postEventReadFromSource(StreamContext const&, ModuleCallingContext const&);
      void postModuleEvent(StreamContext const&, ModuleCallingContext const&);
      void postEvent(StreamContext const&);
      void postEndJob();

      ThreadSafeOutputFileStream file_;
      bool const validFile_; // Separate data member from file to improve efficiency.
      std::chrono::milliseconds const stallThreshold_;
      decltype(now()) beginTime_ {};

      // There can be multiple modules per stream.  Therefore, we need
      // the combination of StreamID and ModuleID to correctly track
      // stalling information.  We use tbb::concurrent_unordered_map
      // for this purpose.
      using StreamID_value = decltype(std::declval<StreamID>().value());
      using ModuleID = decltype(std::declval<ModuleDescription>().id());
      tbb::concurrent_unordered_map<std::pair<StreamID_value,ModuleID>, decltype(beginTime_)> stallStart_ {};

      std::vector<std::string> moduleLabels_ {};
      std::vector<StallStatistics> moduleStats_ {};
    };

  }

}

namespace {
  constexpr char const* filename_default {""};
  constexpr double threshold_default {0.1};
  std::string const space {"  "};
}

using edm::service::StallMonitor;
using namespace std::chrono;

StallMonitor::StallMonitor(ParameterSet const& iPS, ActivityRegistry& iRegistry)
  : file_{iPS.getUntrackedParameter<std::string>("fileName", filename_default)}
  , validFile_{file_}
  , stallThreshold_{static_cast<long int>(iPS.getUntrackedParameter<double>("stallThreshold")*1000)}
{
  iRegistry.watchPreModuleConstruction(this, &StallMonitor::preModuleConstruction);
  iRegistry.watchPostBeginJob(this, &StallMonitor::postBeginJob);
  iRegistry.watchPostModuleEventPrefetching(this, &StallMonitor::postModuleEventPrefetching);
  iRegistry.watchPreModuleEvent(this, &StallMonitor::preModuleEvent);
  iRegistry.watchPostEndJob(this, &StallMonitor::postEndJob);

  if (validFile_) {
    // Only enable the following callbacks if writing to a file.
    iRegistry.watchPreSourceEvent(this, &StallMonitor::preSourceEvent);
    iRegistry.watchPostSourceEvent(this, &StallMonitor::postSourceEvent);
    iRegistry.watchPreEvent(this, &StallMonitor::preEvent);
    iRegistry.watchPreEventReadFromSource(this, &StallMonitor::preEventReadFromSource);
    iRegistry.watchPostEventReadFromSource(this, &StallMonitor::postEventReadFromSource);
    iRegistry.watchPostModuleEvent(this, &StallMonitor::postModuleEvent);
    iRegistry.watchPostEvent(this, &StallMonitor::postEvent);

    std::ostringstream oss;
    oss << "# Step                       Symbol Entries\n"
        << "# -------------------------- ------ ------------------------------------------\n"
        << "# preSourceEvent                " << step::preSourceEvent             << "   <Stream ID> <Time since beginJob (ms)>\n"
        << "# postSourceEvent               " << step::postSourceEvent            << "   <Stream ID> <Time since beginJob (ms)>\n"
        << "# preEvent                      " << step::preEvent                   << "   <Stream ID> <Run#> <LumiBlock#> <Event#> <Time since beginJob (ms)>\n"
        << "# postModuleEventPrefetching    " << step::postModuleEventPrefetching << "   <Stream ID> <Module ID> <Time since beginJob (ms)>\n"
        << "# preModuleEvent                " << step::preModuleEvent             << "   <Stream ID> <Module ID> <Time since beginJob (ms)>\n"
        << "# preEventReadFromSource        " << step::preEventReadFromSource     << "   <Stream ID> <Module ID> <Time since beginJob (ms)>\n"
        << "# postEventReadFromSource       " << step::postEventReadFromSource    << "   <Stream ID> <Module ID> <Time since beginJob (ms)>\n"
        << "# postModuleEvent               " << step::postModuleEvent            << "   <Stream ID> <Module ID> <Time since beginJob (ms)>\n"
        << "# postEvent                     " << step::postEvent                  << "   <Stream ID> <Run#> <LumiBlock#> <Event#> <Time since beginJob (ms)>\n";
    file_.write(oss.str());
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
  // size of moduleLabels_, grow the vector to the correct index and
  // assign the last entry to the desired label.  Note that with the
  // current implementation, there is no module with ID '0'.  In
  // principle, the module-information vectors are therefore each one
  // entry too large.  However, since removing the entry at the front
  // makes for awkward indexing later on, and since the sizes of these
  // extra entries are on the order of bytes, we will leave them in
  // and skip over them later when printing out summaries.  The
  // extraneous entries can be identified by their module labels being
  // empty.
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

  if (validFile_) {
    std::size_t const width {std::to_string(moduleLabels_.size()).size()};

    OStreamColumn col0 {"Module ID", width};
    std::string const lastCol {"Module label"};

    std::ostringstream oss;
    oss << "\n#  " << col0 << space << lastCol << '\n';
    oss << "#  " << std::string(col0.width()+space.size()+lastCol.size(),'-') << '\n';

    for (std::size_t i{} ; i < moduleLabels_.size(); ++i) {
      auto const& label = moduleLabels_[i];
      if (label.empty()) continue; // See comment in filling of moduleLabels_;
      oss << "#M " << std::setw(width) << std::left << col0(i) << space
          << std::left << moduleLabels_[i] << '\n';
    }
    oss << '\n';
    file_.write(oss.str());
  }
  // Don't need the labels anymore--info. is now part of the
  // module-statistics objects.
  moduleLabels_.clear();

  beginTime_ = now();
}

void StallMonitor::preSourceEvent(StreamID const sid)
{
  auto const t = duration_cast<milliseconds>(now()-beginTime_).count();
  auto msg = assembleMessage<step::preSourceEvent>(sid.value(), t);
  file_.write(std::move(msg));
}

void StallMonitor::postSourceEvent(StreamID const sid)
{
  auto const t = duration_cast<milliseconds>(now()-beginTime_).count();
  auto msg = assembleMessage<step::postSourceEvent>(sid.value(), t);
  file_.write(std::move(msg));
}

void StallMonitor::preEvent(StreamContext const& sc)
{
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

  if (validFile_) {
    auto const t = duration_cast<milliseconds>(start-beginTime_).count();
    auto msg = assembleMessage<step::postModuleEventPrefetching>(sid, mid, t);
    file_.write(std::move(msg));
  }
}

void StallMonitor::preModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc)
{
  auto const preModEvent = now();
  auto const sid = stream_id(sc);
  auto const mid = module_id(mcc);
  if (validFile_) {
    auto msg = assembleMessage<step::preModuleEvent>(sid, mid, duration_cast<milliseconds>(preModEvent-beginTime_).count());
    file_.write(std::move(msg));
  }

  auto const preFetch_to_preModEvent = duration_cast<milliseconds>(preModEvent-stallStart_[std::make_pair(sid,mid)]);
  if (preFetch_to_preModEvent < stallThreshold_) return;
  moduleStats_[mid].update(preFetch_to_preModEvent);
}

void StallMonitor::preEventReadFromSource(StreamContext const& sc, ModuleCallingContext const& mcc)
{
  auto const t = duration_cast<milliseconds>(now()-beginTime_).count();
  auto msg = assembleMessage<step::preEventReadFromSource>(stream_id(sc), module_id(mcc), t);
  file_.write(std::move(msg));
}

void StallMonitor::postEventReadFromSource(StreamContext const& sc, ModuleCallingContext const& mcc)
{
  auto const t = duration_cast<milliseconds>(now()-beginTime_).count();
  auto msg = assembleMessage<step::postEventReadFromSource>(stream_id(sc), module_id(mcc), t);
  file_.write(std::move(msg));
}

void StallMonitor::postModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc)
{
  auto const postModEvent = duration_cast<milliseconds>(now()-beginTime_).count();
  auto msg = assembleMessage<step::postModuleEvent>(stream_id(sc), module_id(mcc), postModEvent);
  file_.write(std::move(msg));
}

void StallMonitor::postEvent(StreamContext const& sc)
{
  auto const t = duration_cast<milliseconds>(now()-beginTime_).count();
  auto const& eid = sc.eventID();
  auto msg = assembleMessage<step::postEvent>(stream_id(sc), eid.run(), eid.luminosityBlock(), eid.event(), t);
  file_.write(std::move(msg));
}

void StallMonitor::postEndJob()
{
  // Prepare summary
  std::size_t width {};
  edm::for_all(moduleStats_, [&width](auto const& stats) {
      if (stats.numberOfStalls() == 0u) return;
      width = std::max(width, stats.label().size());
    });

  OStreamColumn tag {"StallMonitor>"};
  OStreamColumn col1 {"Module label", width};
  OStreamColumn col2 {"# of stalls"};
  OStreamColumn col3 {"Total stalled time"};
  OStreamColumn col4 {"Max stalled time"};

  LogAbsolute out {"StallMonitor"};
  out << '\n';
  out << tag << space
      << col1 << space
      << col2 << space
      << col3 << space
      << col4 << '\n';

  out << tag << space
      << std::setfill('-')
      << col1(std::string{}) << space
      << col2(std::string{}) << space
      << col3(std::string{}) << space
      << col4(std::string{}) << '\n';

  using seconds_d = duration<double>;

  auto to_seconds_str = [](auto const& duration){
    std::ostringstream oss;
    auto const time = duration_cast<seconds_d>(duration).count();
    oss << time << " s";
    return oss.str();
  };

  out << std::setfill(' ');
  for (auto const& stats : moduleStats_) {
    if (stats.label().empty() ||  // See comment in filling of moduleLabels_;
        stats.numberOfStalls() == 0u) continue;
    out << std::left
        << tag << space
        << col1(stats.label()) << space
        << std::right
        << col2(stats.numberOfStalls()) << space
        << col3(to_seconds_str(stats.totalStalledTime())) << space
        << col4(to_seconds_str(stats.maxStalledTime())) << '\n';
  }
}

DEFINE_FWK_SERVICE(StallMonitor);
