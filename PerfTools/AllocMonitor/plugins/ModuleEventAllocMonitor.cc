// -*- C++ -*-
//
// Package:     PerfTools/AllocMonitor
// Class  :     ModuleEventAllocMonitor
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Mon, 21 Aug 2023 20:31:57 GMT
//

// system include files
#include <atomic>
#include <numeric>

// user include files
#include "PerfTools/AllocMonitor/interface/AllocMonitorBase.h"
#include "PerfTools/AllocMonitor/interface/AllocMonitorRegistry.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Concurrency/interface/ThreadSafeOutputFileStream.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include "monitor_file_utilities.h"
#include "mea_AllocMap.h"
#include "ThreadTracker.h"

#define DEBUGGER_BREAK

#if defined(DEBUGGER_BREAK)
extern "C" {
void break_on_unmatched_dealloc() {}
}
#endif
namespace {
  using namespace edm::service::moduleEventAlloc;
  using namespace edm::moduleAlloc::monitor_file_utilities;

  struct ThreadAllocInfo {
    AllocMap allocMap_;
    std::vector<void const*> unmatched_;

    //corresponds to temporary memory used
    std::size_t totalMatchedDeallocSize_ = 0;
    //corresponds to memory held over from previous allocation
    std::size_t totalUnmatchedDealloc_ = 0;
    std::size_t numMatchedDeallocs_ = 0;
    std::size_t numUnmatchedDeallocs_ = 0;

    bool active_ = false;
    void alloc(void const* iAddress, std::size_t iSize) { allocMap_.insert(iAddress, iSize); }

    void dealloc(void const* iAddress, std::size_t iSize) {
      auto size = allocMap_.erase(iAddress);
      if (size == 0) {
#if defined(DEBUGGER_BREAK)
        break_on_unmatched_dealloc();
#endif
        totalUnmatchedDealloc_ += iSize;
        ++numUnmatchedDeallocs_;
        unmatched_.push_back(iAddress);
      } else {
        totalMatchedDeallocSize_ += iSize;
        ++numMatchedDeallocs_;
      }
    }

    void reset() {
      totalMatchedDeallocSize_ = 0;
      totalUnmatchedDealloc_ = 0;
      numMatchedDeallocs_ = 0;
      numUnmatchedDeallocs_ = 0;
      allocMap_.clear();
      unmatched_.clear();
      active_ = true;
    }

    void reset(AllocMap const& iBefore) {
      totalMatchedDeallocSize_ = 0;
      totalUnmatchedDealloc_ = 0;
      numMatchedDeallocs_ = 0;
      numUnmatchedDeallocs_ = 0;
      //Need to call this before active_ = true
      allocMap_ = iBefore;
      unmatched_.clear();
      active_ = true;
    }

    void deactivate() { active_ = false; }
  };
  class MonitorAdaptor : public cms::perftools::AllocMonitorBase {
  public:
    static void startOnThread() { threadAllocInfo().reset(); }
    static void startOnThread(AllocMap const& iBefore) { threadAllocInfo().reset(iBefore); }

    static ThreadAllocInfo const& stopOnThread() {
      auto& t = threadAllocInfo();
      if (not t.active_) {
        t.reset();
      } else {
        t.deactivate();
      }
      return t;
    }

  private:
    static ThreadAllocInfo& threadAllocInfo() {
      using namespace cms::perftools::allocMon;
      CMS_THREAD_SAFE static ThreadAllocInfo s_info[ThreadTracker::kTotalEntries];
      return s_info[ThreadTracker::instance().thread_index()];
    }
    void allocCalled(size_t iRequested, size_t iActual, void const* iAddress) final {
      auto& allocInfo = threadAllocInfo();
      if (not allocInfo.active_) {
        return;
      }
      allocInfo.alloc(iAddress, iActual);
    }
    void deallocCalled(size_t iActual, void const* iAddress) final {
      auto& allocInfo = threadAllocInfo();
      if (not allocInfo.active_) {
        return;
      }

      allocInfo.dealloc(iAddress, iActual);
    }
  };

  class Filter {
  public:
    //a negative module id corresponds to an ES module
    Filter(std::vector<int> const* moduleIDs);
    //returns true if should keep this
    //F has an operator() that returns a AllocMap
    template <typename F>
    bool startOnThread(int moduleID, F&&) const;
    const ThreadAllocInfo* stopOnThread(int moduleID) const;

    bool startOnThread() const;
    const ThreadAllocInfo* stopOnThread() const;

    void setGlobalKeep(bool iShouldKeep);
    bool globalKeep() const { return globalKeep_.load(); }

    bool keepModuleInfo(int moduleID) const;

  private:
    std::atomic<bool> globalKeep_ = true;
    std::vector<int> const* moduleIDs_ = nullptr;
  };

  Filter::Filter(std::vector<int> const* moduleIDs) : moduleIDs_{moduleIDs} {}

  template <typename F>
  bool Filter::startOnThread(int moduleID, F&& iInfoFunctor) const {
    if (not globalKeep_.load()) {
      return false;
    }
    if (keepModuleInfo(moduleID)) {
      MonitorAdaptor::startOnThread(iInfoFunctor());
      return true;
    }
    return false;
  }

  const ThreadAllocInfo* Filter::stopOnThread(int moduleID) const {
    if (not globalKeep_.load()) {
      return nullptr;
    }

    if (keepModuleInfo(moduleID)) {
      return &MonitorAdaptor::stopOnThread();
    }
    return nullptr;
  }

  bool Filter::startOnThread() const {
    if (not globalKeep_.load()) {
      return false;
    }
    MonitorAdaptor::startOnThread();
    return true;
  }

  const ThreadAllocInfo* Filter::stopOnThread() const {
    if (not globalKeep_.load()) {
      return nullptr;
    }
    return &MonitorAdaptor::stopOnThread();
  }

  void Filter::setGlobalKeep(bool iShouldKeep) { globalKeep_.store(iShouldKeep); }

  bool Filter::keepModuleInfo(int moduleID) const {
    if ((nullptr == moduleIDs_) or (moduleIDs_->empty()) or
        (std::binary_search(moduleIDs_->begin(), moduleIDs_->end(), moduleID))) {
      return true;
    }
    return false;
  }
}  // namespace

class ModuleEventAllocMonitor {
public:
  ModuleEventAllocMonitor(edm::ParameterSet const& iPS, edm::ActivityRegistry& iAR)
      : moduleNames_(iPS.getUntrackedParameter<std::vector<std::string>>("moduleNames")),
        nEventsToSkip_(iPS.getUntrackedParameter<unsigned int>("nEventsToSkip")),
        filter_(&moduleIDs_) {
    (void)cms::perftools::AllocMonitorRegistry::instance().createAndRegisterMonitor<MonitorAdaptor>();

    if (nEventsToSkip_ > 0) {
      filter_.setGlobalKeep(false);
    }
    auto file = std::make_shared<edm::ThreadSafeOutputFileStream>(iPS.getUntrackedParameter<std::string>("fileName"));
    {
      std::stringstream s;
      s << "#Format\n"
           "# --------\n"
           "# prefixes\n"
           "# #: comment\n"
           "# @: module info\n"
           "# A: memory info for call to 'acquire'\n"
           "# M: memory info for standard module method (i.e. produce, analyze or filter)\n"
           "# D: memory reclaimed when Event products are being deleted at end of Event processing\n"
           "# --------\n"
           "# line formats\n"
           "#@ <module label> <module type> <module ID>\n"
           "#A <module ID> <stream #> <total temp memory (bytes)> <# temp allocations> <total unmatched deallocations "
           "(bytes)> <# unmatched deallocations> <total unmatched allocations [this is copied to #M] (bytes)> <# "
           "unmatched allocations [also copied]>\n"
           "#M <module ID> <stream #> <total temp memory (bytes)> <# temp allocations> <total unmatched deallocations "
           "(bytes)> <# unmatched deallocations> <total unmatched allocations (bytes)> <# unmatched allocations>\n"
           "#D <module ID> <stream #> <total matched deallocations (bytes)> <# matched deallocations>\n";
      file->write(s.str());
    }
    if (not moduleNames_.empty()) {
      iAR.watchPreModuleConstruction([this, file](auto const& description) {
        auto found = std::find(moduleNames_.begin(), moduleNames_.end(), description.moduleLabel());
        if (found != moduleNames_.end()) {
          moduleIDs_.push_back(description.id());
          nModules_ = moduleIDs_.size();
          std::sort(moduleIDs_.begin(), moduleIDs_.end());
          std::stringstream s;
          s << "@ " << description.moduleLabel() << " " << description.moduleName() << " " << description.id() << "\n";
          file->write(s.str());
        }
      });
    } else {
      iAR.watchPreModuleConstruction([this, file](auto const& description) {
        if (description.id() + 1 > nModules_) {
          nModules_ = description.id() + 1;
        }
        std::stringstream s;
        s << "@ " << description.moduleLabel() << " " << description.moduleName() << " " << description.id() << "\n";
        file->write(s.str());
      });
    }
    if (nEventsToSkip_ > 0) {
      iAR.watchPreSourceEvent([this](auto) {
        ++nEventsStarted_;
        if (nEventsStarted_ > nEventsToSkip_) {
          filter_.setGlobalKeep(true);
        }
      });
    }
    iAR.watchPreallocate([this](auto const& alloc) { nStreams_ = alloc.maxNumberOfStreams(); });
    iAR.watchPreBeginJob([this](auto const&, auto const&) {
      streamModuleAllocs_.resize(nStreams_ * nModules_);
      streamModuleInAcquire_ = std::vector<std::atomic<bool>>(nStreams_ * nModules_);
      streamModuleFinishOrder_ = std::vector<int>(nStreams_ * nModules_);
      streamNFinishedModules_ = std::vector<std::atomic<unsigned int>>(nStreams_);
      streamSync_ = std::vector<std::atomic<unsigned int>>(nStreams_);
    });

    iAR.watchPreModuleEvent([this](auto const& iStream, auto const& iMod) {
      auto mod_id = module_id(iMod);
      auto acquireInfo = [this, iStream, mod_id]() {
        //acquire might have started stuff
        streamSync_[iStream.streamID().value()].load();
        auto index = moduleIndex(mod_id);
        auto const& inAcquire = streamModuleInAcquire_[nModules_ * iStream.streamID().value() + index];
        while (inAcquire.load())
          ;
        return streamModuleAllocs_[nModules_ * iStream.streamID().value() + index];
      };
      filter_.startOnThread(mod_id, acquireInfo);
    });
    iAR.watchPostModuleEvent([this, file](auto const& iStream, auto const& iMod) {
      auto mod_id = module_id(iMod);
      auto info = filter_.stopOnThread(mod_id);
      if (info) {
        auto v = std::accumulate(info->allocMap_.allocationSizes().begin(), info->allocMap_.allocationSizes().end(), 0);
        std::stringstream s;
        s << "M " << mod_id << " " << iStream.streamID().value() << " " << info->totalMatchedDeallocSize_ << " "
          << info->numMatchedDeallocs_ << " " << info->totalUnmatchedDealloc_ << " " << info->numUnmatchedDeallocs_
          << " " << v << " " << info->allocMap_.allocationSizes().size() << "\n";
        file->write(s.str());
        auto index = moduleIndex(mod_id);
        auto finishedOrder = streamNFinishedModules_[iStream.streamID().value()]++;
        streamModuleFinishOrder_[finishedOrder + nModules_ * iStream.streamID().value()] =
            nModules_ * iStream.streamID().value() + index;
        streamModuleAllocs_[nModules_ * iStream.streamID().value() + index] = info->allocMap_;
        ++streamSync_[iStream.streamID().value()];
      }
    });

    iAR.watchPreModuleEventAcquire([this](auto const& iStream, auto const& iMod) {
      auto index = moduleIndex(module_id(iMod));
      auto acquireInfo = [index, this, iStream]() {
        streamModuleInAcquire_[nModules_ * iStream.streamID().value() + index].store(true);
        return AllocMap();
      };
      filter_.startOnThread(module_id(iMod), acquireInfo);
    });
    iAR.watchPostModuleEventAcquire([this, file](auto const& iStream, auto const& iMod) {
      auto mod_id = module_id(iMod);
      auto info = filter_.stopOnThread(mod_id);
      if (info) {
        assert(info->allocMap_.allocationSizes().size() == info->allocMap_.size());
        auto v = std::accumulate(info->allocMap_.allocationSizes().begin(), info->allocMap_.allocationSizes().end(), 0);
        std::stringstream s;
        s << "A " << mod_id << " " << iStream.streamID().value() << " " << info->totalMatchedDeallocSize_ << " "
          << info->numMatchedDeallocs_ << " " << info->totalUnmatchedDealloc_ << " " << info->numUnmatchedDeallocs_
          << " " << v << " " << info->allocMap_.allocationSizes().size() << "\n";
        file->write(s.str());
        auto index = mod_id;
        if (not moduleIDs_.empty()) {
          auto it = std::lower_bound(moduleIDs_.begin(), moduleIDs_.end(), mod_id);
          index = it - moduleIDs_.begin();
        }
        {
          auto const& alloc = streamModuleAllocs_[nModules_ * iStream.streamID().value() + index];
          assert(alloc.size() == alloc.allocationSizes().size());
        }
        streamModuleAllocs_[nModules_ * iStream.streamID().value() + index] = info->allocMap_;
        {
          auto const& alloc = streamModuleAllocs_[nModules_ * iStream.streamID().value() + index];
          assert(alloc.size() == alloc.allocationSizes().size());
        }
        ++streamSync_[iStream.streamID().value()];
        streamModuleInAcquire_[nModules_ * iStream.streamID().value() + index].store(false);
      }
    });
    //NOTE: the following watch points may need to be used in the future if allocations occurring during these
    // transition points are confusing the findings.
    /*
    iRegistry.watchPreModuleEventDelayedGet(
                                            StreamEDModuleState<Step::preModuleEventDelayedGet>(logFile, beginTime, iFilter));
    iRegistry.watchPostModuleEventDelayedGet(
                                             StreamEDModuleState<Step::postModuleEventDelayedGet>(logFile, beginTime, iFilter));
    iRegistry.watchPreEventReadFromSource(
                                          StreamEDModuleState<Step::preEventReadFromSource>(logFile, beginTime, iFilter));
    iRegistry.watchPostEventReadFromSource(
                                           StreamEDModuleState<Step::postEventReadFromSource>(logFile, beginTime, iFilter));
    */
    iAR.watchPreClearEvent([this](auto const& iStream) { filter_.startOnThread(); });
    iAR.watchPostClearEvent([this, file](auto const& iStream) {
      auto info = filter_.stopOnThread();
      if (info) {
        streamSync_[iStream.streamID().value()].load();
        //search for associated allocs to deallocs in reverse order that modules finished
        auto nRan = streamNFinishedModules_[iStream.streamID().value()].load();
        auto itBegin = streamModuleFinishOrder_.cbegin() + nModules_ - nRan;
        auto const itEnd = itBegin + nRan;
        streamNFinishedModules_[iStream.streamID().value()].store(0);
        {
          std::vector<std::size_t> moduleDeallocSize(nModules_);
          std::vector<unsigned int> moduleDeallocCount(nModules_);
          for (auto& address : info->unmatched_) {
            decltype(streamModuleAllocs_[0].findOffset(address)) offset;
            auto found = std::find_if(itBegin, itEnd, [&address, &offset, this](auto const& index) {
              auto const& elem = streamModuleAllocs_[index];
              return elem.size() != 0 and (offset = elem.findOffset(address)) != elem.size();
            });
            if (found != itEnd) {
              auto index = *found - nModules_ * iStream.streamID().value();
              moduleDeallocSize[index] += streamModuleAllocs_[*found].allocationSizes()[offset];
              moduleDeallocCount[index] += 1;
            }
          }
          for (unsigned int index = 0; index < nModules_; ++index) {
            if (moduleDeallocCount[index] != 0) {
              auto id = moduleIDs_.empty() ? index : moduleIDs_[index];
              std::stringstream s;
              s << "D " << id << " " << iStream.streamID().value() << " " << moduleDeallocSize[index] << " "
                << moduleDeallocCount[index] << "\n";
              file->write(s.str());
            }
          }
        }

        {
          auto itBegin = streamModuleAllocs_.begin() + nModules_ * iStream.streamID().value();
          auto itEnd = itBegin + nModules_;
          for (auto it = itBegin; it != itEnd; ++it) {
            it->clear();
          }
        }
      }
    });
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& iDesc) {
    edm::ParameterSetDescription ps;
    ps.addUntracked<std::string>("fileName")->setComment("Name of file to write allocation info.");
    ps.addUntracked<std::vector<std::string>>("moduleNames", std::vector<std::string>())
        ->setComment(
            "Module labels for modules which should have their allocations monitored. If empty all modules will be "
            "monitored.");
    ps.addUntracked<unsigned int>("nEventsToSkip", 0)
        ->setComment(
            "Number of events to skip before turning on monitoring. If used in a multi-threaded application, "
            "monitoring may be started for previous events which are still running at the time this threshold is "
            "reached.");
    iDesc.addDefault(ps);
  }

private:
  unsigned int moduleIndex(unsigned int mod_id) const {
    auto index = mod_id;
    if (not moduleIDs_.empty()) {
      auto it = std::lower_bound(moduleIDs_.begin(), moduleIDs_.end(), mod_id);
      index = it - moduleIDs_.begin();
    }
    return index;
  }

  bool forThisModule(unsigned int iID) const {
    return (moduleNames_.empty() or std::binary_search(moduleIDs_.begin(), moduleIDs_.end(), iID));
  }
  //The size is (#streams)*(#modules)
  CMS_THREAD_GUARD(streamSync_) std::vector<AllocMap> streamModuleAllocs_;
  CMS_THREAD_GUARD(streamSync_) std::vector<std::atomic<bool>> streamModuleInAcquire_;
  //This holds the index into the streamModuleAllocs_ for the module which finished
  CMS_THREAD_GUARD(streamSync_) std::vector<int> streamModuleFinishOrder_;
  std::vector<std::atomic<unsigned int>> streamNFinishedModules_;
  std::vector<std::atomic<unsigned int>> streamSync_;
  std::vector<std::string> moduleNames_;
  std::vector<int> moduleIDs_;
  unsigned int nStreams_ = 0;
  unsigned int nModules_ = 0;
  unsigned int nEventsToSkip_ = 0;
  std::atomic<unsigned int> nEventsStarted_{0};
  Filter filter_;
};

DEFINE_FWK_SERVICE(ModuleEventAllocMonitor);
