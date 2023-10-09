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
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/OStreamColumn.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/StdPairHasher.h"
#include "oneapi/tbb/concurrent_unordered_map.h"

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

  using duration_t = std::chrono::microseconds;
  using clock_t = std::chrono::steady_clock;
  auto now = clock_t::now;

  inline auto stream_id(edm::StreamContext const& cs) { return cs.streamID().value(); }

  inline auto module_id(edm::ModuleCallingContext const& mcc) { return mcc.moduleDescription()->id(); }

  inline auto module_id(edm::ESModuleCallingContext const& mcc) { return mcc.componentDescription()->id_; }

  //===============================================================
  class StallStatistics {
  public:
    // c'tor receiving 'std::string const&' type not provided since we
    // must be able to call (e.g.) std::vector<StallStatistics>(20),
    // for which a default label is not sensible in this context.
    StallStatistics() = default;

    std::string const& label() const { return label_; }
    unsigned numberOfStalls() const { return stallCounter_; }

    using rep_t = duration_t::rep;

    duration_t totalStalledTime() const { return duration_t{totalTime_.load()}; }
    duration_t maxStalledTime() const { return duration_t{maxTime_.load()}; }

    // Modifiers
    void setLabel(std::string const& label) { label_ = label; }

    void update(duration_t const ms) {
      ++stallCounter_;
      auto const thisTime = ms.count();
      totalTime_ += thisTime;
      rep_t max{maxTime_};
      while (thisTime > max && !maxTime_.compare_exchange_strong(max, thisTime))
        ;
    }

  private:
    std::string label_{};
    std::atomic<unsigned> stallCounter_{};
    std::atomic<rep_t> totalTime_{};
    std::atomic<rep_t> maxTime_{};
  };

  //===============================================================
  // Message-assembly utilities
  template <typename T>
  std::enable_if_t<std::is_integral<T>::value> concatenate(std::ostream& os, T const t) {
    os << ' ' << t;
  }

  template <typename H, typename... T>
  std::enable_if_t<std::is_integral<H>::value> concatenate(std::ostream& os, H const h, T const... t) {
    os << ' ' << h;
    concatenate(os, t...);
  }

  enum class step : char {
    preSourceEvent = 'S',
    postSourceEvent = 's',
    preEvent = 'E',
    postModuleEventPrefetching = 'p',
    preModuleEventAcquire = 'A',
    postModuleEventAcquire = 'a',
    preModuleEvent = 'M',
    preEventReadFromSource = 'R',
    postEventReadFromSource = 'r',
    postModuleEvent = 'm',
    postEvent = 'e',
    postESModulePrefetching = 'q',
    preESModule = 'N',
    postESModule = 'n',
    preFrameworkTransition = 'F',
    postFrameworkTransition = 'f'
  };

  enum class Phase : short {
    globalEndRun = -4,
    streamEndRun = -3,
    globalEndLumi = -2,
    streamEndLumi = -1,
    Event = 0,
    streamBeginLumi = 1,
    globalBeginLumi = 2,
    streamBeginRun = 3,
    globalBeginRun = 4,
    eventSetupCall = 5
  };

  std::ostream& operator<<(std::ostream& os, step const s) {
    os << static_cast<std::underlying_type_t<step>>(s);
    return os;
  }

  std::ostream& operator<<(std::ostream& os, Phase const s) {
    os << static_cast<std::underlying_type_t<Phase>>(s);
    return os;
  }

  template <step S, typename... ARGS>
  std::string assembleMessage(ARGS const... args) {
    std::ostringstream oss;
    oss << S;
    concatenate(oss, args...);
    oss << '\n';
    return oss.str();
  }

  Phase toTransitionImpl(edm::StreamContext const& iContext) {
    using namespace edm;
    switch (iContext.transition()) {
      case StreamContext::Transition::kBeginRun:
        return Phase::streamBeginRun;
      case StreamContext::Transition::kBeginLuminosityBlock:
        return Phase::streamBeginLumi;
      case StreamContext::Transition::kEvent:
        return Phase::Event;
      case StreamContext::Transition::kEndLuminosityBlock:
        return Phase::streamEndLumi;
      case StreamContext::Transition::kEndRun:
        return Phase::streamEndRun;
      default:
        break;
    }
    assert(false);
    return Phase::Event;
  }

  auto toTransition(edm::StreamContext const& iContext) -> std::underlying_type_t<Phase> {
    return static_cast<std::underlying_type_t<Phase>>(toTransitionImpl(iContext));
  }

  Phase toTransitionImpl(edm::GlobalContext const& iContext) {
    using namespace edm;
    switch (iContext.transition()) {
      case GlobalContext::Transition::kBeginRun:
        return Phase::globalBeginRun;
      case GlobalContext::Transition::kBeginLuminosityBlock:
        return Phase::globalBeginLumi;
      case GlobalContext::Transition::kEndLuminosityBlock:
        return Phase::globalEndLumi;
      case GlobalContext::Transition::kWriteLuminosityBlock:
        return Phase::globalEndLumi;
      case GlobalContext::Transition::kEndRun:
        return Phase::globalEndRun;
      case GlobalContext::Transition::kWriteRun:
        return Phase::globalEndRun;
      default:
        break;
    }
    assert(false);
    return Phase::Event;
  }
  auto toTransition(edm::GlobalContext const& iContext) -> std::underlying_type_t<Phase> {
    return static_cast<std::underlying_type_t<Phase>>(toTransitionImpl(iContext));
  }

}  // namespace

namespace edm {
  namespace service {

    class StallMonitor {
    public:
      StallMonitor(ParameterSet const&, ActivityRegistry&);
      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    private:
      void preModuleConstruction(edm::ModuleDescription const&);
      void preModuleDestruction(edm::ModuleDescription const&);
      void postBeginJob();
      void preSourceEvent(StreamID);
      void postSourceEvent(StreamID);
      void preEvent(StreamContext const&);
      void preModuleEventAcquire(StreamContext const&, ModuleCallingContext const&);
      void postModuleEventAcquire(StreamContext const&, ModuleCallingContext const&);
      void preModuleEvent(StreamContext const&, ModuleCallingContext const&);
      void postModuleEventPrefetching(StreamContext const&, ModuleCallingContext const&);
      void preEventReadFromSource(StreamContext const&, ModuleCallingContext const&);
      void postEventReadFromSource(StreamContext const&, ModuleCallingContext const&);
      void postModuleEvent(StreamContext const&, ModuleCallingContext const&);
      void postEvent(StreamContext const&);
      void preModuleStreamTransition(StreamContext const&, ModuleCallingContext const&);
      void postModuleStreamTransition(StreamContext const&, ModuleCallingContext const&);
      void preModuleGlobalTransition(GlobalContext const&, ModuleCallingContext const&);
      void postModuleGlobalTransition(GlobalContext const&, ModuleCallingContext const&);
      void postEndJob();

      ThreadSafeOutputFileStream file_;
      bool const validFile_;  // Separate data member from file to improve efficiency.
      duration_t const stallThreshold_;
      decltype(now()) beginTime_{};

      // There can be multiple modules per stream.  Therefore, we need
      // the combination of StreamID and ModuleID to correctly track
      // stalling information.  We use oneapi::tbb::concurrent_unordered_map
      // for this purpose.
      using StreamID_value = decltype(std::declval<StreamID>().value());
      using ModuleID = decltype(std::declval<ModuleDescription>().id());
      oneapi::tbb::concurrent_unordered_map<std::pair<StreamID_value, ModuleID>,
                                            std::pair<decltype(beginTime_), bool>,
                                            edm::StdPairHasher>
          stallStart_{};

      std::vector<std::string> moduleLabels_{};
      std::vector<StallStatistics> moduleStats_{};
      std::vector<std::string> esModuleLabels_{};
      unsigned int numStreams_;
    };

  }  // namespace service

}  // namespace edm

namespace {
  constexpr char const* const filename_default{""};
  constexpr double threshold_default{0.1};  //default threashold in seconds
  std::string const space{"  "};
}  // namespace

using edm::service::StallMonitor;
using namespace std::chrono;

StallMonitor::StallMonitor(ParameterSet const& iPS, ActivityRegistry& iRegistry)
    : file_{iPS.getUntrackedParameter<std::string>("fileName")},
      validFile_{file_},
      stallThreshold_{
          std::chrono::round<duration_t>(duration<double>(iPS.getUntrackedParameter<double>("stallThreshold")))} {
  iRegistry.watchPreModuleConstruction(this, &StallMonitor::preModuleConstruction);
  iRegistry.watchPreModuleDestruction(this, &StallMonitor::preModuleDestruction);
  iRegistry.watchPostBeginJob(this, &StallMonitor::postBeginJob);
  iRegistry.watchPostModuleEventPrefetching(this, &StallMonitor::postModuleEventPrefetching);
  iRegistry.watchPreModuleEventAcquire(this, &StallMonitor::preModuleEventAcquire);
  iRegistry.watchPreModuleEvent(this, &StallMonitor::preModuleEvent);
  iRegistry.watchPostEndJob(this, &StallMonitor::postEndJob);

  if (validFile_) {
    // Only enable the following callbacks if writing to a file.
    iRegistry.watchPreSourceEvent(this, &StallMonitor::preSourceEvent);
    iRegistry.watchPostSourceEvent(this, &StallMonitor::postSourceEvent);
    iRegistry.watchPreEvent(this, &StallMonitor::preEvent);
    iRegistry.watchPostModuleEventAcquire(this, &StallMonitor::postModuleEventAcquire);
    iRegistry.watchPreEventReadFromSource(this, &StallMonitor::preEventReadFromSource);
    iRegistry.watchPostEventReadFromSource(this, &StallMonitor::postEventReadFromSource);
    iRegistry.watchPostModuleEvent(this, &StallMonitor::postModuleEvent);
    iRegistry.watchPostEvent(this, &StallMonitor::postEvent);

    iRegistry.watchPreModuleStreamBeginRun(this, &StallMonitor::preModuleStreamTransition);
    iRegistry.watchPostModuleStreamBeginRun(this, &StallMonitor::postModuleStreamTransition);
    iRegistry.watchPreModuleStreamEndRun(this, &StallMonitor::preModuleStreamTransition);
    iRegistry.watchPostModuleStreamEndRun(this, &StallMonitor::postModuleStreamTransition);

    iRegistry.watchPreModuleStreamBeginLumi(this, &StallMonitor::preModuleStreamTransition);
    iRegistry.watchPostModuleStreamBeginLumi(this, &StallMonitor::postModuleStreamTransition);
    iRegistry.watchPreModuleStreamEndLumi(this, &StallMonitor::preModuleStreamTransition);
    iRegistry.watchPostModuleStreamEndLumi(this, &StallMonitor::postModuleStreamTransition);

    iRegistry.watchPreModuleGlobalBeginRun(this, &StallMonitor::preModuleGlobalTransition);
    iRegistry.watchPostModuleGlobalBeginRun(this, &StallMonitor::postModuleGlobalTransition);
    iRegistry.watchPreModuleGlobalEndRun(this, &StallMonitor::preModuleGlobalTransition);
    iRegistry.watchPostModuleGlobalEndRun(this, &StallMonitor::postModuleGlobalTransition);
    iRegistry.watchPreModuleWriteRun(this, &StallMonitor::preModuleGlobalTransition);
    iRegistry.watchPostModuleWriteRun(this, &StallMonitor::postModuleGlobalTransition);

    iRegistry.watchPreModuleGlobalBeginLumi(this, &StallMonitor::preModuleGlobalTransition);
    iRegistry.watchPostModuleGlobalBeginLumi(this, &StallMonitor::postModuleGlobalTransition);
    iRegistry.watchPreModuleGlobalEndLumi(this, &StallMonitor::preModuleGlobalTransition);
    iRegistry.watchPostModuleGlobalEndLumi(this, &StallMonitor::postModuleGlobalTransition);
    iRegistry.watchPreModuleWriteLumi(this, &StallMonitor::preModuleGlobalTransition);
    iRegistry.watchPostModuleWriteLumi(this, &StallMonitor::postModuleGlobalTransition);

    iRegistry.postESModuleRegistrationSignal_.connect([this](auto const& iDescription) {
      if (esModuleLabels_.size() <= iDescription.id_) {
        esModuleLabels_.resize(iDescription.id_ + 1);
      }
      if (not iDescription.label_.empty()) {
        esModuleLabels_[iDescription.id_] = iDescription.label_;
      } else {
        esModuleLabels_[iDescription.id_] = iDescription.type_;
      }
    });

    iRegistry.preESModuleSignal_.connect([this](auto const&, auto const& context) {
      auto const t = duration_cast<duration_t>(now() - beginTime_).count();
      auto msg = assembleMessage<step::preESModule>(
          numStreams_, module_id(context), std::underlying_type_t<Phase>(Phase::eventSetupCall), t);
      file_.write(std::move(msg));
    });
    iRegistry.postESModuleSignal_.connect([this](auto const&, auto const& context) {
      auto const t = duration_cast<duration_t>(now() - beginTime_).count();
      auto msg = assembleMessage<step::postESModule>(
          numStreams_, module_id(context), std::underlying_type_t<Phase>(Phase::eventSetupCall), t);
      file_.write(std::move(msg));
    });

    iRegistry.preallocateSignal_.connect(
        [this](service::SystemBounds const& iBounds) { numStreams_ = iBounds.maxNumberOfStreams(); });

    bool recordFrameworkTransitions = iPS.getUntrackedParameter<bool>("recordFrameworkTransitions");
    if (recordFrameworkTransitions) {
      {
        auto preGlobal = [this](GlobalContext const& gc) {
          auto const t = duration_cast<duration_t>(now() - beginTime_).count();
          auto msg = assembleMessage<step::preFrameworkTransition>(
              numStreams_, gc.luminosityBlockID().run(), gc.luminosityBlockID().luminosityBlock(), toTransition(gc), t);
          file_.write(std::move(msg));
        };
        iRegistry.watchPreGlobalBeginRun(preGlobal);
        iRegistry.watchPreGlobalBeginLumi(preGlobal);
        iRegistry.watchPreGlobalEndLumi(preGlobal);
        iRegistry.watchPreGlobalEndRun(preGlobal);
      }
      {
        auto postGlobal = [this](GlobalContext const& gc) {
          auto const t = duration_cast<duration_t>(now() - beginTime_).count();
          auto msg = assembleMessage<step::postFrameworkTransition>(
              numStreams_, gc.luminosityBlockID().run(), gc.luminosityBlockID().luminosityBlock(), toTransition(gc), t);
          file_.write(std::move(msg));
        };
        iRegistry.watchPostGlobalBeginRun(postGlobal);
        iRegistry.watchPostGlobalBeginLumi(postGlobal);
        iRegistry.watchPostGlobalEndLumi(postGlobal);
        iRegistry.watchPostGlobalEndRun(postGlobal);
      }
      {
        auto preStream = [this](StreamContext const& sc) {
          auto const t = duration_cast<duration_t>(now() - beginTime_).count();
          auto msg = assembleMessage<step::preFrameworkTransition>(
              stream_id(sc), sc.eventID().run(), sc.eventID().luminosityBlock(), toTransition(sc), t);
          file_.write(std::move(msg));
        };
        iRegistry.watchPreStreamBeginRun(preStream);
        iRegistry.watchPreStreamBeginLumi(preStream);
        iRegistry.watchPreStreamEndLumi(preStream);
        iRegistry.watchPreStreamEndRun(preStream);
      }
      {
        auto postStream = [this](StreamContext const& sc) {
          auto const t = duration_cast<duration_t>(now() - beginTime_).count();
          auto msg = assembleMessage<step::postFrameworkTransition>(
              stream_id(sc), sc.eventID().run(), sc.eventID().luminosityBlock(), toTransition(sc), t);
          file_.write(std::move(msg));
        };
        iRegistry.watchPostStreamBeginRun(postStream);
        iRegistry.watchPostStreamBeginLumi(postStream);
        iRegistry.watchPostStreamEndLumi(postStream);
        iRegistry.watchPostStreamEndRun(postStream);
      }
    }

    std::ostringstream oss;
    oss << "# Transition       Symbol\n";
    oss << "#----------------- ------\n";
    oss << "# eventSetupCall  " << Phase::eventSetupCall << "\n"
        << "# globalBeginRun  " << Phase::globalBeginRun << "\n"
        << "# streamBeginRun  " << Phase::streamBeginRun << "\n"
        << "# globalBeginLumi " << Phase::globalBeginLumi << "\n"
        << "# streamBeginLumi " << Phase::streamBeginLumi << "\n"
        << "# Event           " << Phase::Event << "\n"
        << "# streamEndLumi   " << Phase::streamEndLumi << "\n"
        << "# globalEndLumi   " << Phase::globalEndLumi << "\n"
        << "# streamEndRun    " << Phase::streamEndRun << "\n"
        << "# globalEndRun    " << Phase::globalEndRun << "\n";
    oss << "# Step                       Symbol Entries\n"
        << "# -------------------------- ------ ------------------------------------------\n"
        << "# preSourceEvent                " << step::preSourceEvent << "   <Stream ID> <Time since beginJob (ms)>\n"
        << "# postSourceEvent               " << step::postSourceEvent << "   <Stream ID> <Time since beginJob (ms)>\n"
        << "# preEvent                      " << step::preEvent
        << "   <Stream ID> <Run#> <LumiBlock#> <Event#> <Time since beginJob (ms)>\n"
        << "# postModuleEventPrefetching    " << step::postModuleEventPrefetching
        << "   <Stream ID> <Module ID> <Time since beginJob (ms)>\n"
        << "# preModuleEventAcquire         " << step::preModuleEventAcquire
        << "   <Stream ID> <Module ID> <Time since beginJob (ms)>\n"
        << "# postModuleEventAcquire        " << step::postModuleEventAcquire
        << "   <Stream ID> <Module ID> <Time since beginJob (ms)>\n"
        << "# preModuleTransition           " << step::preModuleEvent
        << "   <Stream ID> <Module ID> <Transition type> <Time since beginJob (ms)>\n"
        << "# preEventReadFromSource        " << step::preEventReadFromSource
        << "   <Stream ID> <Module ID> <Time since beginJob (ms)>\n"
        << "# postEventReadFromSource       " << step::postEventReadFromSource
        << "   <Stream ID> <Module ID> <Time since beginJob (ms)>\n"
        << "# postModuleTransition          " << step::postModuleEvent
        << "   <Stream ID> <Module ID> <Transition type> <Time since beginJob (ms)>\n"
        << "# postEvent                     " << step::postEvent
        << "   <Stream ID> <Run#> <LumiBlock#> <Event#> <Time since beginJob (ms)>\n"
        << "# postESModulePrefetching       " << step::postESModulePrefetching
        << "  <Stream ID> <ESModule ID> <Transition type> <Time since beginJob (ms)>\n"
        << "# preESModuleTransition         " << step::preESModule
        << "  <StreamID> <ESModule ID> <TransitionType> <Time since beginJob (ms)>\n"
        << "# postESModuleTransition        " << step::postESModule
        << "  <StreamID> <ESModule ID> <TransitionType> <Time since beginJob (ms)>\n";
    if (recordFrameworkTransitions) {
      oss << "# preFrameworkTransition        " << step::preFrameworkTransition
          << " <Stream ID> <Run#> <LumiBlock#> <Transition type> <Time since beginJob (ms)>\n"
          << "# postFrameworkTransition       " << step::postFrameworkTransition
          << " <Stream ID> <Run#> <LumiBlock#> <Transition type> <Time since beginJob (ms)>\n";
    }
    file_.write(oss.str());
  }
}

void StallMonitor::fillDescriptions(ConfigurationDescriptions& descriptions) {
  ParameterSetDescription desc;
  desc.addUntracked<std::string>("fileName", filename_default)
      ->setComment(
          "Name of file to which detailed timing information should be written.\n"
          "An empty filename argument (the default) indicates that no extra\n"
          "information will be written to a dedicated file, but only the summary\n"
          "including stalling-modules information will be logged.");
  desc.addUntracked<double>("stallThreshold", threshold_default)
      ->setComment(
          "Threshold (in seconds) used to classify modules as stalled.\n"
          "Microsecond granularity allowed.");
  desc.addUntracked<bool>("recordFrameworkTransitions", false)
      ->setComment(
          "When writing a file, include the framework state transitions:\n"
          " stream and global, begin and end, Run and LuminosityBlock.");
  descriptions.add("StallMonitor", desc);
  descriptions.setComment(
      "This service keeps track of various times in event-processing to determine which modules are stalling.");
}

void StallMonitor::preModuleConstruction(ModuleDescription const& md) {
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
  } else {
    moduleLabels_.resize(mid + 1);
    moduleLabels_.back() = md.moduleLabel();
  }
}

void StallMonitor::preModuleDestruction(ModuleDescription const& md) {
  // Reset the module label back if the module is deleted before
  // beginJob() so that the entry is ignored in the summary printouts.
  moduleLabels_[md.id()] = "";
}

void StallMonitor::postBeginJob() {
  // Since a (push,emplace)_back cannot be called for a vector of a
  // type containing atomics (like 'StallStatistics')--i.e. atomics
  // have no copy/move-assignment operators, we must specify the size
  // of the vector at construction time.
  moduleStats_ = std::vector<StallStatistics>(moduleLabels_.size());
  for (std::size_t i{}; i < moduleStats_.size(); ++i) {
    moduleStats_[i].setLabel(moduleLabels_[i]);
  }

  if (validFile_) {
    {
      std::size_t const width{std::to_string(moduleLabels_.size()).size()};
      OStreamColumn col0{"Module ID", width};
      std::string const lastCol{"Module label"};

      std::ostringstream oss;
      oss << "\n#  " << col0 << space << lastCol << '\n';
      oss << "#  " << std::string(col0.width() + space.size() + lastCol.size(), '-') << '\n';

      for (std::size_t i{}; i < moduleLabels_.size(); ++i) {
        auto const& label = moduleLabels_[i];
        if (label.empty())
          continue;  // See comment in filling of moduleLabels_;
        oss << "#M " << std::setw(width) << std::left << col0(i) << space << std::left << moduleLabels_[i] << '\n';
      }
      oss << '\n';
      file_.write(oss.str());
    }
    {
      std::size_t const width{std::to_string(esModuleLabels_.size()).size()};
      OStreamColumn col0{"ESModule ID", width};
      std::string const lastCol{"ESModule label"};

      std::ostringstream oss;
      oss << "\n#  " << col0 << space << lastCol << '\n';
      oss << "#  " << std::string(col0.width() + space.size() + lastCol.size(), '-') << '\n';

      for (std::size_t i{}; i < esModuleLabels_.size(); ++i) {
        auto const& label = esModuleLabels_[i];
        if (label.empty())
          continue;  // See comment in filling of moduleLabels_;
        oss << "#N " << std::setw(width) << std::left << col0(i) << space << std::left << esModuleLabels_[i] << '\n';
      }
      oss << '\n';
      file_.write(oss.str());
    }
  }
  // Don't need the labels anymore--info. is now part of the
  // module-statistics objects.
  decltype(moduleLabels_)().swap(moduleLabels_);
  decltype(esModuleLabels_)().swap(esModuleLabels_);

  beginTime_ = now();
}

void StallMonitor::preSourceEvent(StreamID const sid) {
  auto const t = duration_cast<duration_t>(now() - beginTime_).count();
  auto msg = assembleMessage<step::preSourceEvent>(sid.value(), t);
  file_.write(std::move(msg));
}

void StallMonitor::postSourceEvent(StreamID const sid) {
  auto const t = duration_cast<duration_t>(now() - beginTime_).count();
  auto msg = assembleMessage<step::postSourceEvent>(sid.value(), t);
  file_.write(std::move(msg));
}

void StallMonitor::preEvent(StreamContext const& sc) {
  auto const t = duration_cast<duration_t>(now() - beginTime_).count();
  auto const& eid = sc.eventID();
  auto msg = assembleMessage<step::preEvent>(stream_id(sc), eid.run(), eid.luminosityBlock(), eid.event(), t);
  file_.write(std::move(msg));
}

void StallMonitor::postModuleEventPrefetching(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto const sid = stream_id(sc);
  auto const mid = module_id(mcc);
  auto start = stallStart_[std::make_pair(sid, mid)] = std::make_pair(now(), false);

  if (validFile_) {
    auto const t = duration_cast<duration_t>(start.first - beginTime_).count();
    auto msg = assembleMessage<step::postModuleEventPrefetching>(sid, mid, t);
    file_.write(std::move(msg));
  }
}

void StallMonitor::preModuleEventAcquire(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto const preModEventAcquire = now();
  auto const sid = stream_id(sc);
  auto const mid = module_id(mcc);
  auto& start = stallStart_[std::make_pair(sid, mid)];
  auto startT = start.first.time_since_epoch();
  start.second = true;  // record so the preModuleEvent knows that acquire was called
  if (validFile_) {
    auto t = duration_cast<duration_t>(preModEventAcquire - beginTime_).count();
    auto msg = assembleMessage<step::preModuleEventAcquire>(sid, mid, t);
    file_.write(std::move(msg));
  }
  // Check for stalls if prefetch was called
  if (duration_t::duration::zero() != startT) {
    auto const preFetch_to_preModEventAcquire = duration_cast<duration_t>(preModEventAcquire - start.first);
    if (preFetch_to_preModEventAcquire < stallThreshold_)
      return;
    moduleStats_[mid].update(preFetch_to_preModEventAcquire);
  }
}

void StallMonitor::postModuleEventAcquire(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto const postModEventAcquire = duration_cast<duration_t>(now() - beginTime_).count();
  auto msg = assembleMessage<step::postModuleEventAcquire>(stream_id(sc), module_id(mcc), postModEventAcquire);
  file_.write(std::move(msg));
}

void StallMonitor::preModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto const preModEvent = now();
  auto const sid = stream_id(sc);
  auto const mid = module_id(mcc);
  auto const& start = stallStart_[std::make_pair(sid, mid)];
  auto startT = start.first.time_since_epoch();
  if (validFile_) {
    auto t = duration_cast<duration_t>(preModEvent - beginTime_).count();
    auto msg =
        assembleMessage<step::preModuleEvent>(sid, mid, static_cast<std::underlying_type_t<Phase>>(Phase::Event), t);
    file_.write(std::move(msg));
  }
  // Check for stalls if prefetch was called and we did not already check before acquire
  if (duration_t::duration::zero() != startT && !start.second) {
    auto const preFetch_to_preModEvent = duration_cast<duration_t>(preModEvent - start.first);
    if (preFetch_to_preModEvent < stallThreshold_)
      return;
    moduleStats_[mid].update(preFetch_to_preModEvent);
  }
}

void StallMonitor::preModuleStreamTransition(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto const tNow = now();
  auto const sid = stream_id(sc);
  auto const mid = module_id(mcc);
  auto t = duration_cast<duration_t>(tNow - beginTime_).count();
  auto msg = assembleMessage<step::preModuleEvent>(sid, mid, toTransition(sc), t);
  file_.write(std::move(msg));
}

void StallMonitor::postModuleStreamTransition(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto const t = duration_cast<duration_t>(now() - beginTime_).count();
  auto msg = assembleMessage<step::postModuleEvent>(stream_id(sc), module_id(mcc), toTransition(sc), t);
  file_.write(std::move(msg));
}

void StallMonitor::preModuleGlobalTransition(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  auto t = duration_cast<duration_t>(now() - beginTime_).count();
  auto msg = assembleMessage<step::preModuleEvent>(numStreams_, module_id(mcc), toTransition(gc), t);
  file_.write(std::move(msg));
}

void StallMonitor::postModuleGlobalTransition(GlobalContext const& gc, ModuleCallingContext const& mcc) {
  auto const postModTime = duration_cast<duration_t>(now() - beginTime_).count();
  auto msg = assembleMessage<step::postModuleEvent>(numStreams_, module_id(mcc), toTransition(gc), postModTime);
  file_.write(std::move(msg));
}

void StallMonitor::preEventReadFromSource(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto const t = duration_cast<duration_t>(now() - beginTime_).count();
  auto msg = assembleMessage<step::preEventReadFromSource>(stream_id(sc), module_id(mcc), t);
  file_.write(std::move(msg));
}

void StallMonitor::postEventReadFromSource(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto const t = duration_cast<duration_t>(now() - beginTime_).count();
  auto msg = assembleMessage<step::postEventReadFromSource>(stream_id(sc), module_id(mcc), t);
  file_.write(std::move(msg));
}

void StallMonitor::postModuleEvent(StreamContext const& sc, ModuleCallingContext const& mcc) {
  auto const postModEvent = duration_cast<duration_t>(now() - beginTime_).count();
  auto msg = assembleMessage<step::postModuleEvent>(
      stream_id(sc), module_id(mcc), static_cast<std::underlying_type_t<Phase>>(Phase::Event), postModEvent);
  file_.write(std::move(msg));
}

void StallMonitor::postEvent(StreamContext const& sc) {
  auto const t = duration_cast<duration_t>(now() - beginTime_).count();
  auto const& eid = sc.eventID();
  auto msg = assembleMessage<step::postEvent>(stream_id(sc), eid.run(), eid.luminosityBlock(), eid.event(), t);
  file_.write(std::move(msg));
}

void StallMonitor::postEndJob() {
  // Prepare summary
  std::size_t width{};
  edm::for_all(moduleStats_, [&width](auto const& stats) {
    if (stats.numberOfStalls() == 0u)
      return;
    width = std::max(width, stats.label().size());
  });

  OStreamColumn tag{"StallMonitor>"};
  OStreamColumn col1{"Module label", width};
  OStreamColumn col2{"# of stalls"};
  OStreamColumn col3{"Total stalled time"};
  OStreamColumn col4{"Max stalled time"};

  LogAbsolute out{"StallMonitor"};
  out << '\n';
  out << tag << space << col1 << space << col2 << space << col3 << space << col4 << '\n';

  out << tag << space << std::setfill('-') << col1(std::string{}) << space << col2(std::string{}) << space
      << col3(std::string{}) << space << col4(std::string{}) << '\n';

  using seconds_d = duration<double>;

  auto to_seconds_str = [](auto const& duration) {
    std::ostringstream oss;
    auto const time = duration_cast<seconds_d>(duration).count();
    oss << time << " s";
    return oss.str();
  };

  out << std::setfill(' ');
  for (auto const& stats : moduleStats_) {
    if (stats.label().empty() ||  // See comment in filling of moduleLabels_;
        stats.numberOfStalls() == 0u)
      continue;
    out << std::left << tag << space << col1(stats.label()) << space << std::right << col2(stats.numberOfStalls())
        << space << col3(to_seconds_str(stats.totalStalledTime())) << space
        << col4(to_seconds_str(stats.maxStalledTime())) << '\n';
  }
}

DEFINE_FWK_SERVICE(StallMonitor);
