// -*- C++ -*-
//
// Package:     Services
// Class  :     Timing
//
// Implementation:
//
// Original Author:  Jim Kowalkowski
//

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "FWCore/Utilities/interface/TimingServiceBase.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <iostream>
#include <sstream>
#include <sys/resource.h>
#include <sys/time.h>
#include <atomic>
#include <exception>

namespace edm {

  namespace eventsetup {
    struct ComponentDescription;
    class DataKey;
    class EventSetupRecordKey;
  }  // namespace eventsetup

  namespace service {
    class Timing : public TimingServiceBase {
    public:
      Timing(ParameterSet const&, ActivityRegistry&);
      ~Timing() override;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

      void addToCPUTime(double iTime) override;
      double getTotalCPU() const override;

    private:
      void preBeginJob(PathsAndConsumesOfModulesBase const&, ProcessContext const&);
      void postBeginJob();
      void postEndJob();

      void preEvent(StreamContext const&);
      void postEvent(StreamContext const&);
      void lastPostEvent(double curr_event_time, unsigned int index, StreamContext const& iStream);

      void postModuleEvent(StreamContext const&, ModuleCallingContext const&);

      void preSourceEvent(StreamID);
      void postSourceEvent(StreamID);

      void preSourceLumi(LuminosityBlockIndex);
      void postSourceLumi(LuminosityBlockIndex);

      void preSourceRun(RunIndex);
      void postSourceRun(RunIndex);

      void preOpenFile(std::string const&, bool);
      void postOpenFile(std::string const&, bool);

      void preModule(ModuleDescription const& md);
      void postModule(ModuleDescription const& md);

      void preModuleGlobal(GlobalContext const&, ModuleCallingContext const&);
      void postModuleGlobal(GlobalContext const&, ModuleCallingContext const&);

      void postGlobalBeginRun(GlobalContext const&);
      void postGlobalBeginLumi(GlobalContext const&);

      void preModuleStream(StreamContext const&, ModuleCallingContext const&);
      void postModuleStream(StreamContext const&, ModuleCallingContext const&);

      double postCommon() const;

      struct CountAndTime {
      public:
        CountAndTime(unsigned int count, double time) : count_(count), time_(time) {}
        unsigned int count_;
        double time_;
      };

      void accumulateTimeBegin(std::atomic<CountAndTime*>& countAndTime, double& accumulatedTime);
      void accumulateTimeEnd(std::atomic<CountAndTime*>& countAndTime, double& accumulatedTime);

      double curr_job_time_;               // seconds
      double curr_job_cpu_;                // seconds
      std::atomic<double> extra_job_cpu_;  //seconds
                                           //use last run time for determining end of processing
      std::atomic<double> last_run_time_;
      std::atomic<double> last_run_cpu_;
      std::vector<double> curr_events_time_;  // seconds
      bool summary_only_;
      bool report_summary_;
      double threshold_;
      //
      // Min Max and total event times for each Stream.
      //  Used for summary at end of job
      std::vector<double> max_events_time_;  // seconds
      std::vector<double> min_events_time_;  // seconds
      std::vector<double> sum_events_time_;
      std::atomic<unsigned long> total_event_count_;
      std::atomic<unsigned long> begin_lumi_count_;
      std::atomic<unsigned long> begin_run_count_;
      unsigned int nStreams_;
      unsigned int nThreads_;

      CountAndTime countAndTimeZero_;

      std::atomic<CountAndTime*> countAndTimeForLock_;
      CMS_THREAD_GUARD(countAndTimeForLock_) double accumulatedTimeForLock_;

      std::atomic<CountAndTime*> countAndTimeForGet_;
      CMS_THREAD_GUARD(countAndTimeForGet_) double accumulatedTimeForGet_;

      std::vector<std::unique_ptr<std::atomic<unsigned int>>> countSubProcessesPreEvent_;
      std::vector<std::unique_ptr<std::atomic<unsigned int>>> countSubProcessesPostEvent_;

      bool configuredInTopLevelProcess_;
      unsigned int nSubProcesses_;
    };
  }  // namespace service
}  // namespace edm

namespace edm {
  namespace service {

    static std::string d2str(double d) {
      std::stringstream t;
      t << d;
      return t.str();
    }

    static std::string ui2str(unsigned int i) {
      std::stringstream t;
      t << i;
      return t.str();
    }

    static double getTime() {
      struct timeval t;
      if (gettimeofday(&t, nullptr) < 0)
        throw cms::Exception("SysCallFailed", "Failed call to gettimeofday");
      return static_cast<double>(t.tv_sec) + (static_cast<double>(t.tv_usec) * 1E-6);
    }

    static double getCPU() {
      struct rusage usage;
      getrusage(RUSAGE_SELF, &usage);

      double totalCPUTime = 0.0;
      // User code
      totalCPUTime = (double)usage.ru_utime.tv_sec + (double(usage.ru_utime.tv_usec) * 1E-6);
      // System functions
      totalCPUTime += (double)usage.ru_stime.tv_sec + (double(usage.ru_stime.tv_usec) * 1E-6);

      // Additionally, add in CPU usage from our child processes.
      getrusage(RUSAGE_CHILDREN, &usage);
      totalCPUTime += (double)usage.ru_utime.tv_sec + (double(usage.ru_utime.tv_usec) * 1E-6);
      totalCPUTime += (double)usage.ru_stime.tv_sec + (double(usage.ru_stime.tv_usec) * 1E-6);

      return totalCPUTime;
    }

    //NOTE: We use a per thread stack for module times since unscheduled
    // exectuion or tbb task spawning can cause a module to run on the
    // same thread as an already running module
    static std::vector<double>& moduleTimeStack() {
      static thread_local std::vector<double> s_stack;
      return s_stack;
    }

    static double popStack() {
      auto& modStack = moduleTimeStack();
      assert(!modStack.empty());
      double curr_module_time = modStack.back();
      modStack.pop_back();
      double t = getTime() - curr_module_time;
      return t;
    }

    static void pushStack(bool configuredInTopLevelProcess) {
      if (!configuredInTopLevelProcess) {
        return;
      }
      auto& modStack = moduleTimeStack();
      modStack.push_back(getTime());
    }

    Timing::Timing(ParameterSet const& iPS, ActivityRegistry& iRegistry)
        : curr_job_time_(0.),
          curr_job_cpu_(0.),
          extra_job_cpu_(0.0),
          last_run_time_(0.0),
          last_run_cpu_(0.0),
          curr_events_time_(),
          summary_only_(iPS.getUntrackedParameter<bool>("summaryOnly")),
          report_summary_(iPS.getUntrackedParameter<bool>("useJobReport")),
          threshold_(iPS.getUntrackedParameter<double>("excessiveTimeThreshold")),
          max_events_time_(),
          min_events_time_(),
          total_event_count_(0),
          begin_lumi_count_(0),
          begin_run_count_(0),
          countAndTimeZero_{0, 0.0},
          countAndTimeForLock_{&countAndTimeZero_},
          accumulatedTimeForLock_{0.0},
          countAndTimeForGet_{&countAndTimeZero_},
          accumulatedTimeForGet_{0.0},
          configuredInTopLevelProcess_{false},
          nSubProcesses_{0} {
      iRegistry.watchPreBeginJob(this, &Timing::preBeginJob);
      iRegistry.watchPostBeginJob(this, &Timing::postBeginJob);
      iRegistry.watchPostEndJob(this, &Timing::postEndJob);

      iRegistry.watchPreEvent(this, &Timing::preEvent);
      iRegistry.watchPostEvent(this, &Timing::postEvent);

      bool checkThreshold = true;
      if (threshold_ <= 0.0) {
        //we need to ignore the threshold check
        threshold_ = std::numeric_limits<double>::max();
        checkThreshold = false;
      }

      if ((not summary_only_) || (checkThreshold)) {
        iRegistry.watchPreModuleEvent(this, &Timing::preModuleStream);
        iRegistry.watchPostModuleEvent(this, &Timing::postModuleEvent);
      }
      if (checkThreshold) {
        iRegistry.watchPreSourceEvent(this, &Timing::preSourceEvent);
        iRegistry.watchPostSourceEvent(this, &Timing::postSourceEvent);

        iRegistry.watchPreSourceLumi(this, &Timing::preSourceLumi);
        iRegistry.watchPostSourceLumi(this, &Timing::postSourceLumi);

        iRegistry.watchPreSourceRun(this, &Timing::preSourceRun);
        iRegistry.watchPostSourceRun(this, &Timing::postSourceRun);

        iRegistry.watchPreOpenFile(this, &Timing::preOpenFile);
        iRegistry.watchPostOpenFile(this, &Timing::postOpenFile);

        iRegistry.watchPreEventReadFromSource(this, &Timing::preModuleStream);
        iRegistry.watchPostEventReadFromSource(this, &Timing::postModuleStream);

        iRegistry.watchPreModuleConstruction(this, &Timing::preModule);
        iRegistry.watchPostModuleConstruction(this, &Timing::postModule);

        iRegistry.watchPreModuleBeginJob(this, &Timing::preModule);
        iRegistry.watchPostModuleBeginJob(this, &Timing::postModule);

        iRegistry.watchPreModuleEndJob(this, &Timing::preModule);
        iRegistry.watchPostModuleEndJob(this, &Timing::postModule);

        iRegistry.watchPreModuleStreamBeginRun(this, &Timing::preModuleStream);
        iRegistry.watchPostModuleStreamBeginRun(this, &Timing::postModuleStream);
        iRegistry.watchPreModuleStreamEndRun(this, &Timing::preModuleStream);
        iRegistry.watchPostModuleStreamEndRun(this, &Timing::postModuleStream);

        iRegistry.watchPreModuleStreamBeginLumi(this, &Timing::preModuleStream);
        iRegistry.watchPostModuleStreamBeginLumi(this, &Timing::postModuleStream);
        iRegistry.watchPreModuleStreamEndLumi(this, &Timing::preModuleStream);
        iRegistry.watchPostModuleStreamEndLumi(this, &Timing::postModuleStream);

        iRegistry.watchPreModuleGlobalBeginRun(this, &Timing::preModuleGlobal);
        iRegistry.watchPostModuleGlobalBeginRun(this, &Timing::postModuleGlobal);
        iRegistry.watchPreModuleGlobalEndRun(this, &Timing::preModuleGlobal);
        iRegistry.watchPostModuleGlobalEndRun(this, &Timing::postModuleGlobal);

        iRegistry.watchPreModuleGlobalBeginLumi(this, &Timing::preModuleGlobal);
        iRegistry.watchPostModuleGlobalBeginLumi(this, &Timing::postModuleGlobal);
        iRegistry.watchPreModuleGlobalEndLumi(this, &Timing::preModuleGlobal);
        iRegistry.watchPostModuleGlobalEndLumi(this, &Timing::postModuleGlobal);

        iRegistry.watchPreSourceConstruction(this, &Timing::preModule);
        iRegistry.watchPostSourceConstruction(this, &Timing::postModule);
      }

      iRegistry.watchPostGlobalBeginRun(this, &Timing::postGlobalBeginRun);
      iRegistry.watchPostGlobalBeginLumi(this, &Timing::postGlobalBeginLumi);

      iRegistry.preallocateSignal_.connect([this](service::SystemBounds const& iBounds) {
        nStreams_ = iBounds.maxNumberOfStreams();
        nThreads_ = iBounds.maxNumberOfThreads();
        curr_events_time_.resize(nStreams_, 0.);
        sum_events_time_.resize(nStreams_, 0.);
        max_events_time_.resize(nStreams_, 0.);
        min_events_time_.resize(nStreams_, 1.E6);
        for (unsigned int i = 0; i < nStreams_; ++i) {
          countSubProcessesPreEvent_.emplace_back(std::make_unique<std::atomic<unsigned int>>(0));
          countSubProcessesPostEvent_.emplace_back(std::make_unique<std::atomic<unsigned int>>(0));
        }
      });

      iRegistry.postGlobalEndRunSignal_.connect([this](edm::GlobalContext const&) {
        last_run_time_ = getTime();
        last_run_cpu_ = getCPU();
      });
    }

    Timing::~Timing() {}

    void Timing::addToCPUTime(double iTime) {
      //For accounting purposes we effectively can say we started earlier
      double expected = extra_job_cpu_.load();
      while (not extra_job_cpu_.compare_exchange_strong(expected, expected + iTime)) {
      }
    }

    double Timing::getTotalCPU() const { return getCPU(); }

    void Timing::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.addUntracked<bool>("summaryOnly", false)->setComment("If 'true' do not report timing for each event");
      desc.addUntracked<bool>("useJobReport", true)->setComment("If 'true' write summary information to JobReport");
      desc.addUntracked<double>("excessiveTimeThreshold", 0.)
          ->setComment(
              "Amount of time in seconds before reporting a module or source has taken excessive time. A value of 0.0 "
              "turns off this reporting.");
      descriptions.add("Timing", desc);
      descriptions.setComment("This service reports the time it takes to run each module in a job.");
    }

    void Timing::preBeginJob(PathsAndConsumesOfModulesBase const& pathsAndConsumes, ProcessContext const& pc) {
      if (pc.isSubProcess()) {
        ++nSubProcesses_;
      } else {
        configuredInTopLevelProcess_ = true;
      }
    }

    void Timing::postBeginJob() {
      if (!configuredInTopLevelProcess_) {
        return;
      }
      curr_job_time_ = getTime();
      curr_job_cpu_ = getCPU();

      if (not summary_only_) {
        LogImportant("TimeReport") << "TimeReport> Report activated"
                                   << "\n"
                                   << "TimeReport> Report columns headings for events: "
                                   << "eventnum runnum timetaken\n"
                                   << "TimeReport> Report columns headings for modules: "
                                   << "eventnum runnum modulelabel modulename timetakeni\n"
                                   << "TimeReport> JobTime=" << curr_job_time_ << " JobCPU=" << curr_job_cpu_ << "\n";
      }
    }

    void Timing::postEndJob() {
      if (!configuredInTopLevelProcess_) {
        LogImportant("TimeReport") << "\nTimeReport> This instance of the Timing Service will be disabled because it "
                                      "is configured in a SubProcess.\n"
                                   << "If multiple instances of the TimingService were configured only the one in the "
                                      "top level process will function.\n"
                                   << "The other instance(s) will simply print this message and do nothing.\n\n";
        return;
      }

      const double job_end_time = getTime();
      const double job_end_cpu = getCPU();
      double total_job_time = job_end_time - jobStartTime();

      double total_job_cpu = job_end_cpu + extra_job_cpu_;

      const double total_initialization_time = curr_job_time_ - jobStartTime();
      const double total_initialization_cpu = curr_job_cpu_;

      if (0.0 == jobStartTime()) {
        //did not capture beginning time
        total_job_time = job_end_time - curr_job_time_;
        total_job_cpu = job_end_cpu + extra_job_cpu_ - curr_job_cpu_;
      }

      double min_event_time = *(std::min_element(min_events_time_.begin(), min_events_time_.end()));
      double max_event_time = *(std::max_element(max_events_time_.begin(), max_events_time_.end()));

      auto total_loop_time = last_run_time_ - curr_job_time_;
      auto total_loop_cpu = last_run_cpu_ + extra_job_cpu_ - curr_job_cpu_;

      if (last_run_time_ == 0.0) {
        total_loop_time = 0.0;
        total_loop_cpu = 0.0;
      }

      double sum_all_events_time = 0;
      for (auto t : sum_events_time_) {
        sum_all_events_time += t;
      }

      double average_event_time = 0.0;
      if (total_event_count_ != 0) {
        average_event_time = sum_all_events_time / total_event_count_;
      }

      double event_throughput = 0.0;
      if (total_loop_time != 0.0) {
        event_throughput = total_event_count_ / total_loop_time;
      }

      LogImportant("TimeReport") << "TimeReport> Time report complete in " << total_job_time << " seconds"
                                 << "\n"
                                 << " Time Summary: \n"
                                 << " - Min event:   " << min_event_time << "\n"
                                 << " - Max event:   " << max_event_time << "\n"
                                 << " - Avg event:   " << average_event_time << "\n"
                                 << " - Total loop:  " << total_loop_time << "\n"
                                 << " - Total init:  " << total_initialization_time << "\n"
                                 << " - Total job:   " << total_job_time << "\n"
                                 << " - EventSetup Lock:   " << accumulatedTimeForLock_ << "\n"
                                 << " - EventSetup Get:   " << accumulatedTimeForGet_ << "\n"
                                 << " Event Throughput: " << event_throughput << " ev/s\n"
                                 << " CPU Summary: \n"
                                 << " - Total loop:  " << total_loop_cpu << "\n"
                                 << " - Total init:  " << total_initialization_cpu << "\n"
                                 << " - Total extra: " << extra_job_cpu_ << "\n"
                                 << " - Total job:   " << total_job_cpu << "\n"
                                 << " Processing Summary: \n"
                                 << " - Number of Events:  " << total_event_count_ << "\n"
                                 << " - Number of Global Begin Lumi Calls:  " << begin_lumi_count_ << "\n"
                                 << " - Number of Global Begin Run Calls: " << begin_run_count_ << "\n";

      if (report_summary_) {
        Service<JobReport> reportSvc;
        std::map<std::string, std::string> reportData;

        reportData.insert(std::make_pair("MinEventTime", d2str(min_event_time)));
        reportData.insert(std::make_pair("MaxEventTime", d2str(max_event_time)));
        reportData.insert(std::make_pair("AvgEventTime", d2str(average_event_time)));
        reportData.insert(std::make_pair("EventThroughput", d2str(event_throughput)));
        reportData.insert(std::make_pair("TotalJobTime", d2str(total_job_time)));
        reportData.insert(std::make_pair("TotalJobCPU", d2str(total_job_cpu)));
        reportData.insert(std::make_pair("TotalLoopTime", d2str(total_loop_time)));
        reportData.insert(std::make_pair("TotalLoopCPU", d2str(total_loop_cpu)));
        reportData.insert(std::make_pair("TotalInitTime", d2str(total_initialization_time)));
        reportData.insert(std::make_pair("TotalInitCPU", d2str(total_initialization_cpu)));
        reportData.insert(std::make_pair("NumberOfStreams", ui2str(nStreams_)));
        reportData.insert(std::make_pair("NumberOfThreads", ui2str(nThreads_)));
        reportData.insert(std::make_pair("EventSetup Lock", d2str(accumulatedTimeForLock_)));
        reportData.insert(std::make_pair("EventSetup Get", d2str(accumulatedTimeForGet_)));
        reportSvc->reportPerformanceSummary("Timing", reportData);

        std::map<std::string, std::string> reportData1;
        reportData1.insert(std::make_pair("NumberEvents", ui2str(total_event_count_)));
        reportData1.insert(std::make_pair("NumberBeginLumiCalls", ui2str(begin_lumi_count_)));
        reportData1.insert(std::make_pair("NumberBeginRunCalls", ui2str(begin_run_count_)));
        reportSvc->reportPerformanceSummary("ProcessingSummary", reportData1);
      }
    }

    void Timing::preEvent(StreamContext const& iStream) {
      if (!configuredInTopLevelProcess_) {
        return;
      }
      auto index = iStream.streamID().value();
      if (nSubProcesses_ == 0u) {
        curr_events_time_[index] = getTime();
      } else {
        unsigned int count = ++(*countSubProcessesPreEvent_[index]);
        if (count == 1) {
          curr_events_time_[index] = getTime();
        } else if (count == (nSubProcesses_ + 1)) {
          *countSubProcessesPreEvent_[index] = 0;
        }
      }
    }

    void Timing::postEvent(StreamContext const& iStream) {
      if (!configuredInTopLevelProcess_) {
        return;
      }
      auto index = iStream.streamID().value();
      if (nSubProcesses_ == 0u) {
        lastPostEvent(getTime() - curr_events_time_[index], index, iStream);
      } else {
        unsigned int count = ++(*countSubProcessesPostEvent_[index]);
        if (count == (nSubProcesses_ + 1)) {
          lastPostEvent(getTime() - curr_events_time_[index], index, iStream);
          *countSubProcessesPostEvent_[index] = 0;
        }
      }
    }

    void Timing::lastPostEvent(double curr_event_time, unsigned int index, StreamContext const& iStream) {
      sum_events_time_[index] += curr_event_time;

      if (not summary_only_) {
        auto const& eventID = iStream.eventID();
        LogPrint("TimeEvent") << "TimeEvent> " << eventID.event() << " " << eventID.run() << " " << curr_event_time;
      }
      if (curr_event_time > max_events_time_[index])
        max_events_time_[index] = curr_event_time;
      if (curr_event_time < min_events_time_[index])
        min_events_time_[index] = curr_event_time;
      ++total_event_count_;
    }

    void Timing::postModuleEvent(StreamContext const& iStream, ModuleCallingContext const& iModule) {
      if (!configuredInTopLevelProcess_) {
        return;
      }
      auto const& eventID = iStream.eventID();
      auto const& desc = *(iModule.moduleDescription());
      double t = postCommon();
      if (not summary_only_) {
        LogPrint("TimeModule") << "TimeModule> " << eventID.event() << " " << eventID.run() << " " << desc.moduleLabel()
                               << " " << desc.moduleName() << " " << t;
      }
    }

    void Timing::preSourceEvent(StreamID sid) { pushStack(configuredInTopLevelProcess_); }

    void Timing::postSourceEvent(StreamID sid) { postCommon(); }

    void Timing::preSourceLumi(LuminosityBlockIndex index) { pushStack(configuredInTopLevelProcess_); }

    void Timing::postSourceLumi(LuminosityBlockIndex index) { postCommon(); }

    void Timing::preSourceRun(RunIndex index) { pushStack(configuredInTopLevelProcess_); }

    void Timing::postSourceRun(RunIndex index) { postCommon(); }

    void Timing::preOpenFile(std::string const& lfn, bool b) { pushStack(configuredInTopLevelProcess_); }

    void Timing::postOpenFile(std::string const& lfn, bool b) { postCommon(); }

    void Timing::preModule(ModuleDescription const&) { pushStack(configuredInTopLevelProcess_); }

    void Timing::postModule(ModuleDescription const& desc) { postCommon(); }

    void Timing::preModuleGlobal(GlobalContext const&, ModuleCallingContext const&) {
      pushStack(configuredInTopLevelProcess_);
    }

    void Timing::postModuleGlobal(GlobalContext const&, ModuleCallingContext const& mcc) { postCommon(); }

    void Timing::postGlobalBeginRun(GlobalContext const& gc) {
      if (!configuredInTopLevelProcess_) {
        return;
      }
      if (!gc.processContext()->isSubProcess()) {
        ++begin_run_count_;
      }
    }

    void Timing::postGlobalBeginLumi(GlobalContext const& gc) {
      if (!configuredInTopLevelProcess_) {
        return;
      }
      if (!gc.processContext()->isSubProcess()) {
        ++begin_lumi_count_;
      }
    }

    void Timing::preModuleStream(StreamContext const&, ModuleCallingContext const&) {
      pushStack(configuredInTopLevelProcess_);
    }

    void Timing::postModuleStream(StreamContext const&, ModuleCallingContext const& mcc) { postCommon(); }

    double Timing::postCommon() const {
      if (!configuredInTopLevelProcess_) {
        return 0.0;
      }
      double t = popStack();
      if (t > threshold_) {
        LogError("ExcessiveTime")
            << "ExcessiveTime: Module used " << t
            << " seconds of time which exceeds the error threshold configured in the Timing Service of " << threshold_
            << " seconds.";
      }
      return t;
    }

    void Timing::accumulateTimeBegin(std::atomic<CountAndTime*>& countAndTime, double& accumulatedTime) {
      double newTime = getTime();
      auto newStat = std::make_unique<CountAndTime>(0, newTime);

      CountAndTime* oldStat = countAndTime.load();
      while (true) {
        if (oldStat == nullptr) {
          oldStat = countAndTime.load();
        } else if (countAndTime.compare_exchange_strong(oldStat, nullptr)) {
          break;
        }
      }

      newStat->count_ = oldStat->count_ + 1;
      if (oldStat->count_ != 0) {
        accumulatedTime += (newTime - oldStat->time_) * oldStat->count_;
      }
      countAndTime.store(newStat.release());
      if (oldStat != &countAndTimeZero_) {
        delete oldStat;
      }
    }

    void Timing::accumulateTimeEnd(std::atomic<CountAndTime*>& countAndTime, double& accumulatedTime) {
      double newTime = getTime();

      CountAndTime* oldStat = countAndTime.load();
      while (true) {
        if (oldStat == nullptr) {
          oldStat = countAndTime.load();
        } else if (countAndTime.compare_exchange_strong(oldStat, nullptr)) {
          break;
        }
      }

      if (oldStat->count_ == 1) {
        accumulatedTime += newTime - oldStat->time_;
        countAndTime.store(&countAndTimeZero_);
      } else {
        try {
          auto newStat = std::make_unique<CountAndTime>(oldStat->count_ - 1, newTime);
          accumulatedTime += (newTime - oldStat->time_) * oldStat->count_;
          countAndTime.store(newStat.release());
        } catch (std::exception&) {
          countAndTime.store(oldStat);
          throw;
        }
      }
      delete oldStat;
    }
  }  // namespace service
}  // namespace edm

using edm::service::Timing;

typedef edm::serviceregistry::AllArgsMaker<edm::TimingServiceBase, Timing> TimingMaker;
DEFINE_FWK_SERVICE_MAKER(Timing, TimingMaker);
