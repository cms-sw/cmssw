// -*- C++ -*-
//
// Package:     Services
// Class  :     Memory
//
// Implementation:
//
// Original Author:  Jim Kowalkowski
//
// Change Log
//
// 1 - Apr 25, 2008 M. Fischler
//        Collect event summary information and output to XML file and logger
//        at the end of the job.  Involves split-up of updateAndPrint method.
//
// 2 - May 7, 2008 M. Fischler
//      Collect module summary information and output to XML file and logger
//        at the end of the job.
//
// 3 - Jan 14, 2009 Natalia Garcia Nebot
//        Added:        - Average rate of growth in RSS and peak value attained.
//                - Average rate of growth in VSize over time, Peak VSize
//
//
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/Services/plugins/ProcInfoFetcher.h"

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#ifdef __linux__
#include <malloc.h>
#endif
#include <sstream>
//#include <stdio.h>
#include <string>
//#include <string.h>

#include <cstdio>
#include <atomic>
#include <optional>

// for jemalloc queries
#include <dlfcn.h>
extern "C" {
typedef int (*mallctl_t)(const char* name, void* oldp, size_t* oldlenp, void* newp, size_t newlen);
}

namespace edm {
  namespace service {
    enum class SmapsSection {
      kSharedObject = 0,
      kPcm = 1,
      kOtherFile = 2,
      kStack = 3,
      kMmap = 4,
      kOther = 5,
      kSize = 6
    };
    struct smapsInfo {
      double private_ = 0;        // in MB
      double pss_ = 0;            // in MB
      double anonHugePages_ = 0;  // in MB

      static constexpr auto sectionsSize_ = static_cast<unsigned>(SmapsSection::kSize);
      std::array<double, sectionsSize_> sectionRss_{};    // in MB
      std::array<double, sectionsSize_> sectionVSize_{};  // in MB
    };
    struct JemallocInfo {
      double allocated = 0;  // in MB
      double active = 0;     // in MB
      double resident = 0;   // in MB
      double mapped = 0;     // in MB
      double metadata = 0;   // in MB
    };

    class SimpleMemoryCheck {
    public:
      SimpleMemoryCheck(const ParameterSet&, ActivityRegistry&);
      ~SimpleMemoryCheck();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

      void preSourceConstruction(const ModuleDescription&);
      void postSourceConstruction(const ModuleDescription&);
      void postSourceEvent(StreamID);

      void postBeginJob();

      void postEvent(StreamContext const&);

      void postModuleBeginJob(const ModuleDescription&);
      void postModuleConstruction(const ModuleDescription&);

      void preModule(StreamContext const&, ModuleCallingContext const&);
      void postModule(StreamContext const&, ModuleCallingContext const&);

      void earlyTermination();

      void postEndJob();

      void startSamplingThread();
      void stopSamplingThread();

    private:
      ProcInfo fetch();
      smapsInfo fetchSmaps();
      JemallocInfo fetchJemalloc() const;
      double pageSize() const { return pg_size_; }
      double averageGrowthRate(double current, double past, int count);
      void update();
      void updateMax();
      void andPrintAlways(const std::string& type,
                          const std::string& mdlabel,
                          const std::string& mdname,
                          bool includeSmapsAndJe = false) const;
      void andPrint(const std::string& type, const std::string& mdlabel, const std::string& mdname) const;
      void updateAndPrint(const std::string& type, const std::string& mdlabel, const std::string& mdname);

      // Upon success returns an optional without value
      // Upon failure returns the name of the file the function attempted to open
      std::optional<std::string> openFilesNoThrow();
      void openFiles();

      char const* smapsLineBuffer() const { return get_underlying_safe(smapsLineBuffer_); }
      char*& smapsLineBuffer() { return get_underlying_safe(smapsLineBuffer_); }

      ProcInfo a_;
      ProcInfo b_;
      ProcInfo max_;
      edm::propagate_const<ProcInfo*> current_;
      edm::propagate_const<ProcInfo*> previous_;

      smapsInfo currentSmaps_;

      ProcInfoFetcher piFetcher_;
      double pg_size_;
      int num_to_skip_;
      //options
      bool showMallocInfo_;
      bool showJemallocInfo_;
      bool oncePerEventMode_;
      bool printEachTime_;
      bool jobReportOutputOnly_;
      bool monitorPssAndPrivate_;
      std::atomic<int> count_;
      unsigned int sampleEveryNSeconds_;
      std::optional<std::thread> samplingThread_;
      std::atomic<bool> stopThread_ = false;
      std::atomic<edm::EventID> mostRecentlyStartedEvent_;

      mallctl_t je_mallctl = nullptr;

      //smaps
      edm::propagate_const<FILE*> smapsFile_ = nullptr;
      edm::propagate_const<char*> smapsLineBuffer_;
      size_t smapsLineBufferLen_;

      //Rates of growth
      double growthRateVsize_;
      double growthRateRss_;

      // Event summary statistics 				changeLog 1
      struct SignificantEvent {
        edm::EventID event;
        double vsize = 0;
        double deltaVsize = 0;
        double rss = 0;
        double deltaRss = 0;
        double privateSize = 0;
        double pss = 0;
        double anonHugePages = 0;
        std::optional<JemallocInfo> jemalloc;
        int count = 0;
        bool monitorPssAndPrivate = false;
        SignificantEvent() = default;
        void set(double deltaV, double deltaR, edm::EventID const& e, SimpleMemoryCheck* t) {
          count = t->count_;
          vsize = t->current_->vsize;
          deltaVsize = deltaV;
          rss = t->current_->rss;
          deltaRss = deltaR;
          monitorPssAndPrivate = t->monitorPssAndPrivate_;
          if (monitorPssAndPrivate) {
            privateSize = t->currentSmaps_.private_;
            pss = t->currentSmaps_.pss_;
            anonHugePages = t->currentSmaps_.anonHugePages_;
          }
          if (t->showJemallocInfo_) {
            jemalloc = t->fetchJemalloc();
          }
          event = e;
        }
      };  // SignificantEvent
      friend struct SignificantEvent;
      friend std::ostream& operator<<(std::ostream& os, SimpleMemoryCheck::SignificantEvent const& se);

      /*
       Significative events for deltaVsize:
       - eventM_: Event which makes the biggest value for deltaVsize
       - eventL1_: Event which makes the second biggest value for deltaVsize
       - eventL2_: Event which makes the third biggest value for deltaVsize
       - eventR1_: Event which makes the second biggest value for deltaVsize
       - eventR2_: Event which makes the third biggest value for deltaVsize
       M>L1>L2 and M>R1>R2
       Unknown relation between Ls and Rs events ???????
       Significative events for vsize:
       - eventT1_: Event whith the biggest value for vsize
       - eventT2_: Event whith the second biggest value for vsize
       - eventT3_: Event whith the third biggest value for vsize
       T1>T2>T3
       */
      SignificantEvent eventM_;
      SignificantEvent eventL1_;
      SignificantEvent eventL2_;
      SignificantEvent eventR1_;
      SignificantEvent eventR2_;
      SignificantEvent eventT1_;
      SignificantEvent eventT2_;
      SignificantEvent eventT3_;

      /*
       Significative event for deltaRss:
       - eventRssT1_: Event whith the biggest value for rss
       - eventRssT2_: Event whith the second biggest value for rss
       - eventRssT3_: Event whith the third biggest value for rss
       T1>T2>T3
       Significative events for deltaRss:
       - eventDeltaRssT1_: Event whith the biggest value for deltaRss
       - eventDeltaRssT2_: Event whith the second biggest value for deltaRss
       - eventDeltaRssT3_: Event whith the third biggest value for deltaRss
       T1>T2>T3
       */
      SignificantEvent eventRssT1_;
      SignificantEvent eventRssT2_;
      SignificantEvent eventRssT3_;
      SignificantEvent eventDeltaRssT1_;
      SignificantEvent eventDeltaRssT2_;
      SignificantEvent eventDeltaRssT3_;

      void updateEventStats(edm::EventID const& e);
      std::string eventStatOutput(std::string title, SignificantEvent const& e) const;
      void eventStatOutput(std::string title, SignificantEvent const& e, std::map<std::string, std::string>& m) const;
      std::string mallOutput(std::string title, size_t const& n) const;

      // Module summary statistices
      struct SignificantModule {
        int postEarlyCount;
        double totalDeltaVsize;
        double maxDeltaVsize;
        edm::EventID eventMaxDeltaV;
        double totalEarlyVsize;
        double maxEarlyVsize;
        SignificantModule()
            : postEarlyCount(0),
              totalDeltaVsize(0),
              maxDeltaVsize(0),
              eventMaxDeltaV(),
              totalEarlyVsize(0),
              maxEarlyVsize(0) {}
        void set(double deltaV, bool early);
      };  // SignificantModule
      friend struct SignificantModule;
      friend std::ostream& operator<<(std::ostream& os, SimpleMemoryCheck::SignificantModule const& se);
      bool moduleSummaryRequested_;
      typedef std::map<std::string, SignificantModule> SignificantModulesMap;
      SignificantModulesMap modules_;
      double moduleEntryVsize_;
      void updateModuleMemoryStats(SignificantModule& m, double dv, edm::EventID const&);

      //Used to guarantee we only do one measurement at a time
      std::atomic<bool> measurementUnderway_;
      std::atomic<bool> moduleMeasurementUnderway_;
      std::atomic<unsigned int> moduleStreamID_;
      std::atomic<unsigned int> moduleID_;

    };  // SimpleMemoryCheck

    std::ostream& operator<<(std::ostream& os, SimpleMemoryCheck::SignificantEvent const& se);

    std::ostream& operator<<(std::ostream& os, SimpleMemoryCheck::SignificantModule const& se);

  }  // namespace service
}  // namespace edm

#ifdef __linux__
#define LINUX 1
#endif

#include <fcntl.h>
#include <unistd.h>

namespace edm {
  namespace service {

    static std::string d2str(double d) {
      std::ostringstream t;
      t << d;
      return t.str();
    }

    static std::string i2str(int i) {
      std::ostringstream t;
      t << i;
      return t.str();
    }

    ProcInfo SimpleMemoryCheck::fetch() { return piFetcher_.fetch(); }

    smapsInfo SimpleMemoryCheck::fetchSmaps() {
      smapsInfo ret;
#ifdef LINUX
      fseek(smapsFile_, 0, SEEK_SET);
      ssize_t read;
      SmapsSection section = SmapsSection::kOther;

      /*
       The format of the report is
       Private_Clean:        0 kB
       Private_Dirty:       72 kB
       Swap:                 0 kB
       Pss:                 72 kB
       AnonHugePages:    10240 kB
       */

      while ((read = getline(&smapsLineBuffer(), &smapsLineBufferLen_, smapsFile_)) != -1) {
        if (read > 14) {
          // Are we in a line that defines a mapping?
          // (a character following ':' is not a space)
          if (char const* ret = strchr(smapsLineBuffer_, ':'); ret != nullptr and *(ret + 1) != ' ') {
            ret = strrchr(smapsLineBuffer_, ' ');
            if (ret == nullptr) {
              // shouldn't happen, but let's protect anyway
              section = SmapsSection::kOther;
            } else if (*(ret + 1) == '\n') {
              // no "path" element
              section = SmapsSection::kMmap;
            } else if (*(ret + 1) == '/') {
              // "path" starts with '/', assume it's file
              // differentiate shared object and .pcm files
              auto len = strlen(ret);
              if (0 == strncmp(ret + len - 5, ".pcm", 4)) {
                section = SmapsSection::kPcm;
              } else if (strstr(ret, ".so") != nullptr) {
                section = SmapsSection::kSharedObject;
              } else {
                section = SmapsSection::kOtherFile;
              }
            } else if (0 == strncmp("[stack]", ret + 1, 7)) {
              section = SmapsSection::kStack;
            } else {
              section = SmapsSection::kOther;
            }
            continue;
          }

          //Private
          if (0 == strncmp("Private_", smapsLineBuffer_, 8)) {
            unsigned int value = atoi(smapsLineBuffer_ + 14);
            //Convert from kB to MB
            ret.private_ += static_cast<double>(value) / 1024.;
          } else if (0 == strncmp("Pss:", smapsLineBuffer_, 4)) {
            unsigned int value = atoi(smapsLineBuffer_ + 4);
            //Convert from kB to MB
            ret.pss_ += static_cast<double>(value) / 1024.;
          } else if (0 == strncmp("AnonHugePages:", smapsLineBuffer_, 14)) {
            unsigned int value = atoi(smapsLineBuffer_ + 14);
            ret.anonHugePages_ += static_cast<double>(value) / 1024.;
          } else if (0 == strncmp("Rss:", smapsLineBuffer_, 4)) {
            unsigned int value = atoi(smapsLineBuffer_ + 4);
            //Convert from kB to MB
            ret.sectionRss_[static_cast<unsigned>(section)] += static_cast<double>(value) / 1024.;
          } else if (0 == strncmp("Size:", smapsLineBuffer_, 5)) {
            unsigned int value = atoi(smapsLineBuffer_ + 5);
            //Convert from kB to MB
            ret.sectionVSize_[static_cast<unsigned>(section)] += static_cast<double>(value) / 1024.;
          }
        }
      }
#endif
      return ret;
    }

    JemallocInfo SimpleMemoryCheck::fetchJemalloc() const {
      JemallocInfo info;
      if (je_mallctl) {
        // refresh stats
        uint64_t epoch = 1;
        size_t e_len = sizeof(uint64_t);
        if (je_mallctl("epoch", &epoch, &e_len, &epoch, e_len) != 0) {
          return info;
        }

        // query values
        size_t allocated, active, resident, mapped, metadata;
        size_t len = sizeof(size_t);
        if (je_mallctl("stats.allocated", &allocated, &len, nullptr, 0) != 0) {
          return info;
        }
        if (je_mallctl("stats.active", &active, &len, nullptr, 0) != 0) {
          return info;
        }
        if (je_mallctl("stats.resident", &resident, &len, nullptr, 0) != 0) {
          return info;
        }
        if (je_mallctl("stats.mapped", &mapped, &len, nullptr, 0) != 0) {
          return info;
        }
        if (je_mallctl("stats.metadata", &metadata, &len, nullptr, 0) != 0) {
          return info;
        }
        info.allocated = allocated / 1024.0 / 1024.0;
        info.active = active / 1024.0 / 1024.0;
        info.resident = resident / 1024.0 / 1024.0;
        info.mapped = mapped / 1024.0 / 1024.0;
        info.metadata = metadata / 1024.0 / 1024.0;
      }
      return info;
    }

    double SimpleMemoryCheck::averageGrowthRate(double current, double past, int count) {
      return (current - past) / (double)count;
    }

    SimpleMemoryCheck::SimpleMemoryCheck(ParameterSet const& iPS, ActivityRegistry& iReg)
        : a_(),
          b_(),
          current_(&a_),
          previous_(&b_),
          pg_size_(sysconf(_SC_PAGESIZE)),  // getpagesize()
          num_to_skip_(iPS.getUntrackedParameter<int>("ignoreTotal")),
          showMallocInfo_(iPS.getUntrackedParameter<bool>("showMallocInfo")),
          showJemallocInfo_(iPS.getUntrackedParameter<bool>("showJemallocInfo")),
          oncePerEventMode_(iPS.getUntrackedParameter<bool>("oncePerEventMode")),
          printEachTime_(oncePerEventMode_ or iPS.getUntrackedParameter<bool>("printEachSample")),
          jobReportOutputOnly_(iPS.getUntrackedParameter<bool>("jobReportOutputOnly")),
          monitorPssAndPrivate_(iPS.getUntrackedParameter<bool>("monitorPssAndPrivate")),
          count_(),
          sampleEveryNSeconds_(iPS.getUntrackedParameter<unsigned int>("sampleEveryNSeconds")),
          smapsFile_(nullptr),
          smapsLineBuffer_(nullptr),
          smapsLineBufferLen_(0),
          growthRateVsize_(),
          growthRateRss_(),
          moduleSummaryRequested_(iPS.getUntrackedParameter<bool>("moduleMemorySummary")),
          measurementUnderway_(false) {
      // changelog 2
      // pg_size = (double)getpagesize();
      std::ostringstream ost;

      if (monitorPssAndPrivate_) {
        openFiles();
      }

      iReg.watchPostEndJob(this, &SimpleMemoryCheck::postEndJob);
      // A possible source for early termination is a signal from WM
      // when the job's memory use exceeds their limit
      iReg.watchPreSourceEarlyTermination([this](TerminationOrigin) { earlyTermination(); });
      iReg.watchPreGlobalEarlyTermination([this](GlobalContext const&, TerminationOrigin) { earlyTermination(); });
      iReg.watchPreStreamEarlyTermination([this](StreamContext const&, TerminationOrigin) { earlyTermination(); });

      if (sampleEveryNSeconds_ > 0) {
        if (oncePerEventMode_) {
          throw edm::Exception(edm::errors::Configuration)
              << "'sampleEventNSeconds' and 'oncePerEventMode' cannot be used together";
        }
        if (moduleSummaryRequested_) {
          throw edm::Exception(edm::errors::Configuration)
              << "'sampleEventNSeconds' and 'moduleSummaryRequested' cannot be used together";
        }
        iReg.watchPostBeginJob(this, &SimpleMemoryCheck::startSamplingThread);
        iReg.watchPreEndJob(this, &SimpleMemoryCheck::stopSamplingThread);
        iReg.watchPreEvent([this](auto const& iContext) { mostRecentlyStartedEvent_.store(iContext.eventID()); });
        return;
      }

      iReg.watchPostEvent(this, &SimpleMemoryCheck::postEvent);

      if (!oncePerEventMode_) {  // default, prints on increases
        iReg.watchPreSourceConstruction(this, &SimpleMemoryCheck::preSourceConstruction);
        iReg.watchPostSourceConstruction(this, &SimpleMemoryCheck::postSourceConstruction);
        iReg.watchPostSourceEvent(this, &SimpleMemoryCheck::postSourceEvent);
        iReg.watchPostModuleConstruction(this, &SimpleMemoryCheck::postModuleConstruction);
        iReg.watchPostModuleBeginJob(this, &SimpleMemoryCheck::postModuleBeginJob);
        iReg.watchPostModuleEvent(this, &SimpleMemoryCheck::postModule);
        iReg.watchPostBeginJob(this, &SimpleMemoryCheck::postBeginJob);
      }
      if (moduleSummaryRequested_) {  // changelog 2
        iReg.watchPreModuleEvent(this, &SimpleMemoryCheck::preModule);
        if (oncePerEventMode_) {
          iReg.watchPostModuleEvent(this, &SimpleMemoryCheck::postModule);
        }
      }

      // The following are not currenty used/implemented below for either
      // of the print modes (but are left here for reference)
      //  iReg.watchPostBeginJob(this,
      //       &SimpleMemoryCheck::postBeginJob);
      //  iReg.watchPreProcessEvent(this,
      //       &SimpleMemoryCheck::preEventProcessing);
      //  iReg.watchPreModule(this,
      //       &SimpleMemoryCheck::preModule);

      if (showJemallocInfo_) {
        // jemalloc's mallctl(), if we use jemalloc
        je_mallctl = reinterpret_cast<mallctl_t>(::dlsym(RTLD_DEFAULT, "mallctl"));
        if (je_mallctl == nullptr) {
          showJemallocInfo_ = false;
        }
      }
    }

    SimpleMemoryCheck::~SimpleMemoryCheck() {
#ifdef LINUX
      if (nullptr != smapsFile_) {
        fclose(smapsFile_);
      }
#endif
      if (smapsLineBuffer_) {
        //getline will create the memory using malloc
        free(smapsLineBuffer_);
      }
    }

    void SimpleMemoryCheck::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.addUntracked<int>("ignoreTotal", 1)
          ->setComment("Number of events/samples to finish before starting measuring and reporting.");
      desc.addUntracked<unsigned int>("sampleEveryNSeconds", 0)
          ->setComment(
              "Use a special thread to sample memory at the set rate. A value of 0 means no sampling. This option "
              "cannot be used with 'oncePerEventMode' or 'moduleMemorySummary'.");
      desc.addUntracked<bool>("printEachSample", false)
          ->setComment("If sampling on, print each sample taken else will print only when sample is the largest seen.");
      desc.addUntracked<bool>("showMallocInfo", false);
      desc.addUntracked<bool>("showJemallocInfo", true)
          ->setComment(
              "If enabled and jemalloc is being used, print high-level jemalloc statistics at the early termination "
              "and endJob printouts as well as for the peak VSIZE and RSS -using records.");
      desc.addUntracked<bool>("oncePerEventMode", false)
          ->setComment(
              "Only check memory at the end of each event. Not as useful in multi-threaded job as other running events "
              "contribute.");
      desc.addUntracked<bool>("jobReportOutputOnly", false);
      desc.addUntracked<bool>("monitorPssAndPrivate", false);
      desc.addUntracked<bool>("moduleMemorySummary", false)
          ->setComment(
              "Track significant memory events for each module. This does not work well in multi-threaded jobs.");
      descriptions.add("SimpleMemoryCheck", desc);
    }

    std::optional<std::string> SimpleMemoryCheck::openFilesNoThrow() {
#ifdef LINUX
      std::ostringstream smapsNameOst;
      smapsNameOst << "/proc/" << getpid() << "/smaps";
      auto smapsName = smapsNameOst.str();
      if ((smapsFile_ = fopen(smapsName.c_str(), "r")) == nullptr) {
        return smapsName;
      }
#endif
      return {};
    }

    void SimpleMemoryCheck::openFiles() {
      auto smapsFileNameIfFailed = openFilesNoThrow();
      if (smapsFileNameIfFailed.has_value()) {
        throw Exception(errors::Configuration) << "Failed to open smaps file " << *smapsFileNameIfFailed << std::endl;
      }
    }

    void SimpleMemoryCheck::postBeginJob() {
      growthRateVsize_ = current_->vsize;
      growthRateRss_ = current_->rss;
    }

    void SimpleMemoryCheck::preSourceConstruction(ModuleDescription const& md) {
      bool expected = false;
      if (measurementUnderway_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        std::shared_ptr<void> guard(
            nullptr, [this](void const*) { measurementUnderway_.store(false, std::memory_order_release); });
        updateAndPrint("pre-ctor", md.moduleLabel(), md.moduleName());
      }
    }

    void SimpleMemoryCheck::postSourceConstruction(ModuleDescription const& md) {
      bool expected = false;
      if (measurementUnderway_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        std::shared_ptr<void> guard(
            nullptr, [this](void const*) { measurementUnderway_.store(false, std::memory_order_release); });
        updateAndPrint("ctor", md.moduleLabel(), md.moduleName());
      }
    }

    void SimpleMemoryCheck::postSourceEvent(StreamID sid) {
      bool expected = false;
      if (measurementUnderway_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        std::shared_ptr<void> guard(
            nullptr, [this](void const*) { measurementUnderway_.store(false, std::memory_order_release); });
        updateAndPrint("module", "source", "source");
      }
    }

    void SimpleMemoryCheck::postModuleConstruction(ModuleDescription const& md) {
      bool expected = false;
      if (measurementUnderway_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        std::shared_ptr<void> guard(
            nullptr, [this](void const*) { measurementUnderway_.store(false, std::memory_order_release); });
        updateAndPrint("ctor", md.moduleLabel(), md.moduleName());
      }
    }

    void SimpleMemoryCheck::postModuleBeginJob(ModuleDescription const& md) {
      bool expected = false;
      if (measurementUnderway_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        std::shared_ptr<void> guard(
            nullptr, [this](void const*) { measurementUnderway_.store(false, std::memory_order_release); });
        updateAndPrint("beginJob", md.moduleLabel(), md.moduleName());
      }
    }

    void SimpleMemoryCheck::earlyTermination() {
      bool expected = false;
      while (not measurementUnderway_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        expected = false;
      }
      std::shared_ptr<void> guard(
          nullptr, [this](void const*) { measurementUnderway_.store(false, std::memory_order_release); });
      if (not smapsFile_) {
        openFilesNoThrow();
      }
      if (smapsFile_) {
        currentSmaps_ = fetchSmaps();
      }
      update();
      andPrintAlways("earlyTermination", "", "", true);
      updateMax();
    }

    void SimpleMemoryCheck::startSamplingThread() {
      samplingThread_ = std::thread{[this]() {
        while (not stopThread_) {
          std::this_thread::sleep_for(std::chrono::duration<unsigned int>(sampleEveryNSeconds_));
          ++count_;
          update();
          if (monitorPssAndPrivate_) {
            currentSmaps_ = fetchSmaps();
          }
          auto e = mostRecentlyStartedEvent_.load();
          andPrint("sampling", "", "");
          updateEventStats(e);
          updateMax();
        }
      }};
    }
    void SimpleMemoryCheck::stopSamplingThread() {
      stopThread_ = true;
      samplingThread_->join();
    }

    void SimpleMemoryCheck::postEndJob() {
      if (not jobReportOutputOnly_) {
        LogAbsolute log("MemoryReport");

        update();
        log << "MemoryReport> EndJob: virtual size " << current_->vsize << " Mbytes, RSS " << current_->rss
            << " Mbytes";
        // extract smaps information if file open succeeded
        if (not smapsFile_) {
          openFilesNoThrow();
        }
        if (smapsFile_) {
          currentSmaps_ = fetchSmaps();
          auto soRss = currentSmaps_.sectionRss_[static_cast<unsigned>(SmapsSection::kSharedObject)];
          auto pcmRss = currentSmaps_.sectionRss_[static_cast<unsigned>(SmapsSection::kPcm)];
          auto otherFileRss = currentSmaps_.sectionRss_[static_cast<unsigned>(SmapsSection::kOtherFile)];
          auto mmapRss = currentSmaps_.sectionRss_[static_cast<unsigned>(SmapsSection::kMmap)];
          auto soVSize = currentSmaps_.sectionVSize_[static_cast<unsigned>(SmapsSection::kSharedObject)];
          auto pcmVSize = currentSmaps_.sectionVSize_[static_cast<unsigned>(SmapsSection::kPcm)];
          auto otherFileVSize = currentSmaps_.sectionVSize_[static_cast<unsigned>(SmapsSection::kOtherFile)];
          auto mmapVSize = currentSmaps_.sectionVSize_[static_cast<unsigned>(SmapsSection::kMmap)];
          log << ", PSS " << currentSmaps_.pss_ << " MBytes, Private " << currentSmaps_.private_ << "\n AnonHugePages "
              << currentSmaps_.anonHugePages_ << " Mbytes\n"
              << " mmapped memory pages " << mmapVSize << " Mbytes (VSize), " << mmapRss << " MBytes (RSS)\n"
              << " mmapped file pages " << (soVSize + pcmVSize + otherFileVSize) << " Mbytes (VSize), "
              << (soRss + pcmRss + otherFileRss) << " MBytes (RSS)\n"
              << "  of which .so's " << soVSize << " Mbytes (VSize), " << soRss << " MBytes (RSS)\n"
              << "  of which PCM's " << pcmVSize << " Mbytes (VSize), " << pcmRss << " MBytes (RSS)\n"
              << "  of which other " << otherFileVSize << " Mbytes (VSize), " << otherFileRss << " MBytes (RSS)";
        }
        if (showJemallocInfo_) {
          auto info = fetchJemalloc();
          log << "\n Jemalloc allocated " << info.allocated << " MBytes, active " << info.active
              << " MBytes\n  resident " << info.resident << " Mbytes, mapped " << info.mapped << " Mbytes\n  metadata "
              << info.metadata << " Mbytes";
        }
        log << "\n";

        auto logJemalloc = [&log](std::optional<JemallocInfo> const& info) {
          if (info.has_value()) {
            log << "\n Jemalloc allocated " << info->allocated << " active " << info->active << " resident "
                << info->resident << " mapped " << info->mapped << " metadata " << info->metadata;
          }
        };

        log << "MemoryReport> Peak virtual size " << eventT1_.vsize << " Mbytes (RSS " << eventT1_.rss << ")";
        logJemalloc(eventT1_.jemalloc);
        log << "\n Key events increasing vsize: \n"
            << eventL2_ << "\n"
            << eventL1_ << "\n"
            << eventM_ << "\n"
            << eventR1_ << "\n"
            << eventR2_ << "\n"
            << eventT3_ << "\n"
            << eventT2_ << "\n"
            << eventT1_ << "\nMemoryReport> Peak rss size " << eventRssT1_.rss << " Mbytes (VSIZE " << eventRssT1_.vsize
            << ")";
        ;
        logJemalloc(eventRssT1_.jemalloc);
        log << "\n Key events increasing rss:\n"
            << eventRssT3_ << "\n"
            << eventRssT2_ << "\n"
            << eventRssT1_ << "\n"
            << eventDeltaRssT3_ << "\n"
            << eventDeltaRssT2_ << "\n"
            << eventDeltaRssT1_;
      }
      if (moduleSummaryRequested_ and not jobReportOutputOnly_) {  // changelog 1
        LogAbsolute mmr("ModuleMemoryReport");                     // at end of if block, mmr
                                                                   // is destructed, causing
                                                                   // message to be logged
        mmr << "ModuleMemoryReport> Each line has module label and: \n";
        mmr << "  (after early ignored events) \n";
        mmr << "    count of times module executed; average increase in vsize \n";
        mmr << "    maximum increase in vsize; event on which maximum occurred \n";
        mmr << "  (during early ignored events) \n";
        mmr << "    total and maximum vsize increases \n \n";
        for (SignificantModulesMap::iterator im = modules_.begin(); im != modules_.end(); ++im) {
          SignificantModule const& m = im->second;
          if (m.totalDeltaVsize == 0 && m.totalEarlyVsize == 0)
            continue;
          mmr << im->first << ": ";
          mmr << "n = " << m.postEarlyCount;
          if (m.postEarlyCount > 0) {
            mmr << " avg = " << m.totalDeltaVsize / m.postEarlyCount;
          }
          mmr << " max = " << m.maxDeltaVsize << " " << m.eventMaxDeltaV;
          if (m.totalEarlyVsize > 0) {
            mmr << " early total: " << m.totalEarlyVsize;
            mmr << " max: " << m.maxEarlyVsize;
          }
          mmr << "\n";
        }
      }  // end of if; mmr goes out of scope; log message is queued

      Service<JobReport> reportSvc;
      // changelog 1
#define SIMPLE_MEMORY_CHECK_ORIGINAL_XML_OUTPUT
#ifdef SIMPLE_MEMORY_CHECK_ORIGINAL_XML_OUTPUT
      //     std::map<std::string, double> reportData;
      std::map<std::string, std::string> reportData;

      if (eventL2_.vsize > 0)
        eventStatOutput("LargeVsizeIncreaseEventL2", eventL2_, reportData);
      if (eventL1_.vsize > 0)
        eventStatOutput("LargeVsizeIncreaseEventL1", eventL1_, reportData);
      if (eventM_.vsize > 0)
        eventStatOutput("LargestVsizeIncreaseEvent", eventM_, reportData);
      if (eventR1_.vsize > 0)
        eventStatOutput("LargeVsizeIncreaseEventR1", eventR1_, reportData);
      if (eventR2_.vsize > 0)
        eventStatOutput("LargeVsizeIncreaseEventR2", eventR2_, reportData);
      if (eventT3_.vsize > 0)
        eventStatOutput("ThirdLargestVsizeEventT3", eventT3_, reportData);
      if (eventT2_.vsize > 0)
        eventStatOutput("SecondLargestVsizeEventT2", eventT2_, reportData);
      if (eventT1_.vsize > 0)
        eventStatOutput("LargestVsizeEventT1", eventT1_, reportData);

      if (eventRssT3_.rss > 0)
        eventStatOutput("ThirdLargestRssEvent", eventRssT3_, reportData);
      if (eventRssT2_.rss > 0)
        eventStatOutput("SecondLargestRssEvent", eventRssT2_, reportData);
      if (eventRssT1_.rss > 0)
        eventStatOutput("LargestRssEvent", eventRssT1_, reportData);
      if (eventDeltaRssT3_.deltaRss > 0)
        eventStatOutput("ThirdLargestIncreaseRssEvent", eventDeltaRssT3_, reportData);
      if (eventDeltaRssT2_.deltaRss > 0)
        eventStatOutput("SecondLargestIncreaseRssEvent", eventDeltaRssT2_, reportData);
      if (eventDeltaRssT1_.deltaRss > 0)
        eventStatOutput("LargestIncreaseRssEvent", eventDeltaRssT1_, reportData);

#ifdef __linux__
#if (__GLIBC__ > 2) || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33)
      struct mallinfo2 minfo = mallinfo2();
#else
      struct mallinfo minfo = mallinfo();
#endif
      reportData.insert(std::make_pair("HEAP_ARENA_SIZE_BYTES", std::to_string(minfo.arena)));
      reportData.insert(std::make_pair("HEAP_ARENA_N_UNUSED_CHUNKS", std::to_string(minfo.ordblks)));
      reportData.insert(std::make_pair("HEAP_TOP_FREE_BYTES", std::to_string(minfo.keepcost)));
      reportData.insert(std::make_pair("HEAP_MAPPED_SIZE_BYTES", std::to_string(minfo.hblkhd)));
      reportData.insert(std::make_pair("HEAP_MAPPED_N_CHUNKS", std::to_string(minfo.hblks)));
      reportData.insert(std::make_pair("HEAP_USED_BYTES", std::to_string(minfo.uordblks)));
      reportData.insert(std::make_pair("HEAP_UNUSED_BYTES", std::to_string(minfo.fordblks)));
#endif

      // Report Growth rates for VSize and Rss
      reportData.insert(std::make_pair("AverageGrowthRateVsize",
                                       d2str(averageGrowthRate(current_->vsize, growthRateVsize_, count_))));
      reportData.insert(std::make_pair("PeakValueVsize", d2str(eventT1_.vsize)));
      reportData.insert(
          std::make_pair("AverageGrowthRateRss", d2str(averageGrowthRate(current_->rss, growthRateRss_, count_))));
      reportData.insert(std::make_pair("PeakValueRss", d2str(eventRssT1_.rss)));

      if (moduleSummaryRequested_) {  // changelog 2
        for (SignificantModulesMap::iterator im = modules_.begin(); im != modules_.end(); ++im) {
          SignificantModule const& m = im->second;
          if (m.totalDeltaVsize == 0 && m.totalEarlyVsize == 0)
            continue;
          std::string label = im->first + ":";
          reportData.insert(std::make_pair(label + "PostEarlyCount", i2str(m.postEarlyCount)));
          if (m.postEarlyCount > 0) {
            reportData.insert(std::make_pair(label + "AverageDeltaVsize", d2str(m.totalDeltaVsize / m.postEarlyCount)));
          }
          reportData.insert(std::make_pair(label + "MaxDeltaVsize", d2str(m.maxDeltaVsize)));
          if (m.totalEarlyVsize > 0) {
            reportData.insert(std::make_pair(label + "TotalEarlyVsize", d2str(m.totalEarlyVsize)));
            reportData.insert(std::make_pair(label + "MaxEarlyDeltaVsize", d2str(m.maxEarlyVsize)));
          }
        }
      }

      std::map<std::string, std::string> reportMemoryProperties;

      if (FILE* fmeminfo = fopen("/proc/meminfo", "r")) {
        char buf[128];
        char space[] = " ";
        size_t value;
        while (fgets(buf, sizeof(buf), fmeminfo)) {
          char* saveptr;
          char* token = nullptr;
          token = strtok_r(buf, space, &saveptr);
          if (token != nullptr) {
            value = atol(strtok_r(nullptr, space, &saveptr));
            std::string category = token;
            reportMemoryProperties.insert(std::make_pair(category.substr(0, strlen(token) - 1), i2str(value)));
          }
        }

        fclose(fmeminfo);
      }

      //      reportSvc->reportMemoryInfo(reportData, reportMemoryProperties);
      reportSvc->reportPerformanceSummary("ApplicationMemory", reportData);
      reportSvc->reportPerformanceSummary("SystemMemory", reportMemoryProperties);
#endif

#ifdef SIMPLE_MEMORY_CHECK_DIFFERENT_XML_OUTPUT
      std::vector<std::string> reportData;

      if (eventL2_.vsize > 0)
        reportData.push_back(eventStatOutput("LargeVsizeIncreaseEventL2", eventL2_));
      if (eventL1_.vsize > 0)
        reportData.push_back(eventStatOutput("LargeVsizeIncreaseEventL1", eventL1_));
      if (eventM_.vsize > 0)
        reportData.push_back(eventStatOutput("LargestVsizeIncreaseEvent", eventM_));
      if (eventR1_.vsize > 0)
        reportData.push_back(eventStatOutput("LargeVsizeIncreaseEventR1", eventR1_));
      if (eventR2_.vsize > 0)
        reportData.push_back(eventStatOutput("LargeVsizeIncreaseEventR2", eventR2_));
      if (eventT3_.vsize > 0)
        reportData.push_back(eventStatOutput("ThirdLargestVsizeEventT3", eventT3_));
      if (eventT2_.vsize > 0)
        reportData.push_back(eventStatOutput("SecondLargestVsizeEventT2", eventT2_));
      if (eventT1_.vsize > 0)
        reportData.push_back(eventStatOutput("LargestVsizeEventT1", eventT1_));

      if (eventRssT3_.rss > 0)
        eventStatOutput("ThirdLargestRssEvent", eventRssT3_, reportData);
      if (eventRssT2_.rss > 0)
        eventStatOutput("SecondLargestRssEvent", eventRssT2_, reportData);
      if (eventRssT1_.rss > 0)
        eventStatOutput("LargestRssEvent", eventRssT1_, reportData);
      if (eventDeltaRssT3_.deltaRss > 0)
        eventStatOutput("ThirdLargestIncreaseRssEvent", eventDeltaRssT3_, reportData);
      if (eventDeltaRssT2_.deltaRss > 0)
        eventStatOutput("SecondLargestIncreaseRssEvent", eventDeltaRssT2_, reportData);
      if (eventDeltaRssT1_.deltaRss > 0)
        eventStatOutput("LargestIncreaseRssEvent", eventDeltaRssT1_, reportData);

#if (__GLIBC__ > 2) || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33)
      struct mallinfo2 minfo = mallinfo2();
#else
      struct mallinfo minfo = mallinfo();
#endif
      reportData.push_back(mallOutput("HEAP_ARENA_SIZE_BYTES", minfo.arena));
      reportData.push_back(mallOutput("HEAP_ARENA_N_UNUSED_CHUNKS", minfo.ordblks));
      reportData.push_back(mallOutput("HEAP_TOP_FREE_BYTES", minfo.keepcost));
      reportData.push_back(mallOutput("HEAP_MAPPED_SIZE_BYTES", minfo.hblkhd));
      reportData.push_back(mallOutput("HEAP_MAPPED_N_CHUNKS", minfo.hblks));
      reportData.push_back(mallOutput("HEAP_USED_BYTES", minfo.uordblks));
      reportData.push_back(mallOutput("HEAP_UNUSED_BYTES", minfo.fordblks));

      // Report Growth rates for VSize and Rss
      reportData.insert(std::make_pair("AverageGrowthRateVsize",
                                       d2str(averageGrowthRate(current_->vsize, growthRateVsize_, count_))));
      reportData.insert(std::make_pair("PeakValueVsize", d2str(eventT1_.vsize)));
      reportData.insert(
          std::make_pair("AverageGrowthRateRss", d2str(averageGrowthRate(current_->rss, growthRateRss_, count_))));
      reportData.insert(std::make_pair("PeakValueRss", d2str(eventRssT1_.rss)));

      reportSvc->reportMemoryInfo(reportData);
      // This is a form of reportMemoryInfo taking s vector, not a map
#endif
    }  // postEndJob

    void SimpleMemoryCheck::postEvent(StreamContext const& iContext) {
      ++count_;
      bool expected = false;
      if (measurementUnderway_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        std::shared_ptr<void> guard(
            nullptr, [this](void const*) { measurementUnderway_.store(false, std::memory_order_release); });
        update();
        if (monitorPssAndPrivate_) {
          currentSmaps_ = fetchSmaps();
        }
        updateEventStats(iContext.eventID());
        if (oncePerEventMode_) {
          // should probably use be Run:Event or count_ for the label and name
          updateMax();
          andPrint("event", "", "");
        }
      }
    }

    void SimpleMemoryCheck::preModule(StreamContext const& iStreamContext, ModuleCallingContext const& iModuleContext) {
      bool expectedMeasurementUnderway = false;
      if (measurementUnderway_.compare_exchange_strong(expectedMeasurementUnderway, true, std::memory_order_acq_rel)) {
        std::shared_ptr<void> guard(
            nullptr, [this](void const*) { measurementUnderway_.store(false, std::memory_order_release); });
        bool expectedModuleMeasurementUnderway = false;
        if (moduleMeasurementUnderway_.compare_exchange_strong(expectedModuleMeasurementUnderway, true)) {
          update();
          // changelog 2
          moduleEntryVsize_ = current_->vsize;
          moduleStreamID_.store(iStreamContext.streamID().value(), std::memory_order_release);
          moduleID_.store(iModuleContext.moduleDescription()->id(), std::memory_order_release);
        }
      }
    }

    void SimpleMemoryCheck::postModule(StreamContext const& iStreamContext,
                                       ModuleCallingContext const& iModuleContext) {
      if (!oncePerEventMode_) {
        bool expected = false;
        if (measurementUnderway_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
          std::shared_ptr<void> guard(
              nullptr, [this](void const*) { measurementUnderway_.store(false, std::memory_order_release); });
          auto const md = iModuleContext.moduleDescription();
          updateAndPrint("module", md->moduleLabel(), md->moduleName());
        }
      }

      if (moduleSummaryRequested_) {
        //is this the module instance we are measuring?
        if (moduleMeasurementUnderway_.load(std::memory_order_acquire) and
            (iStreamContext.streamID().value() == moduleStreamID_.load(std::memory_order_acquire)) and
            (iModuleContext.moduleDescription()->id() == moduleID_.load(std::memory_order_acquire))) {
          //Need to release our module measurement lock
          std::shared_ptr<void> guardModuleMeasurementUnderway(
              nullptr, [this](void const*) { moduleMeasurementUnderway_.store(false, std::memory_order_release); });
          bool expected = false;
          if (measurementUnderway_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            std::shared_ptr<void> guardMeasurementUnderway(
                nullptr, [this](void const*) { measurementUnderway_.store(false, std::memory_order_release); });
            if (oncePerEventMode_) {
              update();
            }
            // changelog 2
            double dv = current_->vsize - moduleEntryVsize_;
            std::string label = iModuleContext.moduleDescription()->moduleLabel();
            updateModuleMemoryStats(modules_[label], dv, iStreamContext.eventID());
          }
        }
      }
    }

    void SimpleMemoryCheck::update() {
      std::swap(current_, previous_);
      *current_ = fetch();
    }

    void SimpleMemoryCheck::updateMax() {
      auto v = *current_;
      if ((v > max_) || oncePerEventMode_) {
        if (max_.vsize < v.vsize) {
          max_.vsize = v.vsize;
        }
        if (max_.rss < v.rss) {
          max_.rss = v.rss;
        }
      }
    }

    void SimpleMemoryCheck::updateEventStats(EventID const& e) {
      if (count_ < num_to_skip_)
        return;
      if (count_ == num_to_skip_) {
        eventT1_.set(0, 0, e, this);
        eventM_.set(0, 0, e, this);
        eventRssT1_.set(0, 0, e, this);
        eventDeltaRssT1_.set(0, 0, e, this);
        return;
      }
      double vsize = current_->vsize;
      double deltaVsize = vsize - eventT1_.vsize;

      // Update significative events for Vsize
      if (vsize > eventT1_.vsize) {
        double deltaRss = current_->rss - eventT1_.rss;
        eventT3_ = eventT2_;
        eventT2_ = eventT1_;
        eventT1_.set(deltaVsize, deltaRss, e, this);
      } else if (vsize > eventT2_.vsize) {
        double deltaRss = current_->rss - eventT1_.rss;
        eventT3_ = eventT2_;
        eventT2_.set(deltaVsize, deltaRss, e, this);
      } else if (vsize > eventT3_.vsize) {
        double deltaRss = current_->rss - eventT1_.rss;
        eventT3_.set(deltaVsize, deltaRss, e, this);
      }

      if (deltaVsize > eventM_.deltaVsize) {
        double deltaRss = current_->rss - eventM_.rss;
        if (eventL1_.deltaVsize >= eventR1_.deltaVsize) {
          eventL2_ = eventL1_;
        } else {
          eventL2_ = eventR1_;
        }
        eventL1_ = eventM_;
        eventM_.set(deltaVsize, deltaRss, e, this);
        eventR1_ = SignificantEvent();
        eventR2_ = SignificantEvent();
      } else if (deltaVsize > eventR1_.deltaVsize) {
        double deltaRss = current_->rss - eventM_.rss;
        eventR2_ = eventR1_;
        eventR1_.set(deltaVsize, deltaRss, e, this);
      } else if (deltaVsize > eventR2_.deltaVsize) {
        double deltaRss = current_->rss - eventR1_.rss;
        eventR2_.set(deltaVsize, deltaRss, e, this);
      }

      // Update significative events for Rss
      double rss = current_->rss;
      double deltaRss = rss - eventRssT1_.rss;

      if (rss > eventRssT1_.rss) {
        eventRssT3_ = eventRssT2_;
        eventRssT2_ = eventRssT1_;
        eventRssT1_.set(deltaVsize, deltaRss, e, this);
      } else if (rss > eventRssT2_.rss) {
        eventRssT3_ = eventRssT2_;
        eventRssT2_.set(deltaVsize, deltaRss, e, this);
      } else if (rss > eventRssT3_.rss) {
        eventRssT3_.set(deltaVsize, deltaRss, e, this);
      }
      if (deltaRss > eventDeltaRssT1_.deltaRss) {
        eventDeltaRssT3_ = eventDeltaRssT2_;
        eventDeltaRssT2_ = eventDeltaRssT1_;
        eventDeltaRssT1_.set(deltaVsize, deltaRss, e, this);
      } else if (deltaRss > eventDeltaRssT2_.deltaRss) {
        eventDeltaRssT3_ = eventDeltaRssT2_;
        eventDeltaRssT2_.set(deltaVsize, deltaRss, e, this);
      } else if (deltaRss > eventDeltaRssT3_.deltaRss) {
        eventDeltaRssT3_.set(deltaVsize, deltaRss, e, this);
      }
    }  // updateEventStats

    void SimpleMemoryCheck::andPrintAlways(std::string const& type,
                                           std::string const& mdlabel,
                                           std::string const& mdname,
                                           bool includeSmapsAndJe) const {
      double deltaVSIZE = current_->vsize - max_.vsize;
      double deltaRSS = current_->rss - max_.rss;

      LogWarning log("MemoryCheck");
      // default
      log << "MemoryCheck: " << type << " " << mdname << ":" << mdlabel << " VSIZE " << current_->vsize << " "
          << deltaVSIZE << " RSS " << current_->rss << " " << deltaRSS;
      if (showMallocInfo_) {
#ifdef __linux__
#if (__GLIBC__ > 2) || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33)
        struct mallinfo2 minfo = mallinfo2();
#else
        struct mallinfo minfo = mallinfo();
#endif
        log << " HEAP-ARENA [ SIZE-BYTES " << minfo.arena << " N-UNUSED-CHUNKS " << minfo.ordblks << " TOP-FREE-BYTES "
            << minfo.keepcost << " ]"
            << " HEAP-MAPPED [ SIZE-BYTES " << minfo.hblkhd << " N-CHUNKS " << minfo.hblks << " ]"
            << " HEAP-USED-BYTES " << minfo.uordblks << " HEAP-UNUSED-BYTES " << minfo.fordblks;
#endif
      }
      if (includeSmapsAndJe) {
        if (smapsFile_) {
          log << " PSS " << currentSmaps_.pss_ << " PRIVATE " << currentSmaps_.private_ << " ANONHUGEPAGES "
              << currentSmaps_.anonHugePages_;
        }
        if (je_mallctl) {
          auto info = fetchJemalloc();
          log << " JeMalloc allocated " << info.allocated << " active " << info.active << " resident " << info.resident
              << " mapped " << info.mapped << " metadata " << info.metadata;
        }
      }
    }

    void SimpleMemoryCheck::andPrint(std::string const& type,
                                     std::string const& mdlabel,
                                     std::string const& mdname) const {
      if (not jobReportOutputOnly_ && ((*current_ > max_) || printEachTime_)) {
        if (count_ >= num_to_skip_) {
          andPrintAlways(type, mdlabel, mdname);
        }
      }
    }

    void SimpleMemoryCheck::updateAndPrint(std::string const& type,
                                           std::string const& mdlabel,
                                           std::string const& mdname) {
      update();
      andPrint(type, mdlabel, mdname);
      updateMax();
    }

#ifdef SIMPLE_MEMORY_CHECK_ORIGINAL_XML_OUTPUT
    void SimpleMemoryCheck::eventStatOutput(std::string title,
                                            SignificantEvent const& e,
                                            std::map<std::string, std::string>& m) const {
      {
        std::ostringstream os;
        os << title << "-a-COUNT";
        m.insert(std::make_pair(os.str(), i2str(e.count)));
      }
      {
        std::ostringstream os;
        os << title << "-b-RUN";
        m.insert(std::make_pair(os.str(), d2str(static_cast<double>(e.event.run()))));
      }
      {
        std::ostringstream os;
        os << title << "-c-EVENT";
        m.insert(std::make_pair(os.str(), d2str(static_cast<double>(e.event.event()))));
      }
      {
        std::ostringstream os;
        os << title << "-d-VSIZE";
        m.insert(std::make_pair(os.str(), d2str(e.vsize)));
      }
      {
        std::ostringstream os;
        os << title << "-e-DELTV";
        m.insert(std::make_pair(os.str(), d2str(e.deltaVsize)));
      }
      {
        std::ostringstream os;
        os << title << "-f-RSS";
        m.insert(std::make_pair(os.str(), d2str(e.rss)));
      }
      if (monitorPssAndPrivate_) {
        {
          std::ostringstream os;
          os << title << "-g-PRIVATE";
          m.insert(std::make_pair(os.str(), d2str(e.privateSize)));
        }
        {
          std::ostringstream os;
          os << title << "-h-PSS";
          m.insert(std::make_pair(os.str(), d2str(e.pss)));
        }
      }
    }  // eventStatOutput
#endif

#ifdef SIMPLE_MEMORY_CHECK_DIFFERENT_XML_OUTPUT
    std::string SimpleMemoryCheck::eventStatOutput(std::string title, SignificantEvent const& e) const {
      std::ostringstream os;
      os << "  <" << title << ">\n";
      os << "    " << e.count << ": " << e.event;
      os << " vsize " << e.vsize - e.deltaVsize << " + " << e.deltaVsize << " = " << e.vsize;
      os << "  rss: " << e.rss << "\n";
      os << "  </" << title << ">\n";
      return os.str();
    }  // eventStatOutput

    std::string SimpleMemoryCheck::mallOutput(std::string title, size_t const& n) const {
      std::ostringstream os;
      os << "  <" << title << ">\n";
      os << "    " << n << "\n";
      os << "  </" << title << ">\n";
      return os.str();
    }
#endif
    // changelog 2
    void SimpleMemoryCheck::updateModuleMemoryStats(SignificantModule& m,
                                                    double dv,
                                                    edm::EventID const& currentEventID) {
      if (count_ < num_to_skip_) {
        m.totalEarlyVsize += dv;
        if (dv > m.maxEarlyVsize)
          m.maxEarlyVsize = dv;
      } else {
        ++m.postEarlyCount;
        m.totalDeltaVsize += dv;
        if (dv > m.maxDeltaVsize) {
          m.maxDeltaVsize = dv;
          m.eventMaxDeltaV = currentEventID;
        }
      }
    }  //updateModuleMemoryStats

    std::ostream& operator<<(std::ostream& os, SimpleMemoryCheck::SignificantEvent const& se) {
      os << "[" << se.count << "] " << se.event << "  vsize = " << se.vsize << " deltaVsize = " << se.deltaVsize
         << " rss = " << se.rss << " delta = " << se.deltaRss;

      if (se.monitorPssAndPrivate) {
        os << " private = " << se.privateSize << " pss = " << se.pss;
      }
      if (se.jemalloc.has_value()) {
        os << " allocated = " << se.jemalloc->allocated << " active = " << se.jemalloc->active;
      }
      return os;
    }

    std::ostream& operator<<(std::ostream& os, SimpleMemoryCheck::SignificantModule const& sm) {
      if (sm.postEarlyCount > 0) {
        os << "\nPost Early Events:  TotalDeltaVsize: " << sm.totalDeltaVsize
           << " (avg: " << sm.totalDeltaVsize / sm.postEarlyCount << "; max: " << sm.maxDeltaVsize << " during "
           << sm.eventMaxDeltaV << ")";
      }
      if (sm.totalEarlyVsize > 0) {
        os << "\n     Early Events:  TotalDeltaVsize: " << sm.totalEarlyVsize << " (max: " << sm.maxEarlyVsize << ")";
      }

      return os;
    }

  }  // end namespace service
}  // end namespace edm

#if defined(__linux__)
using edm::service::SimpleMemoryCheck;
DEFINE_FWK_SERVICE(SimpleMemoryCheck);
#endif
