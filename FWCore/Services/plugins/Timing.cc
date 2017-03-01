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
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <sstream>
#include <sys/resource.h>
#include <sys/time.h>
#include <atomic>

namespace edm {
  
  namespace service {
    class Timing : public TimingServiceBase {
    public:
      Timing(ParameterSet const&, ActivityRegistry&);
      ~Timing();
      
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      
      virtual void addToCPUTime(StreamID id, double iTime) override;
      virtual double getTotalCPU() const override;
      
    private:
      
      void postBeginJob();
      void postEndJob();
      
      void preEvent(StreamContext const&);
      void postEvent(StreamContext const&);
      
      void preModule(StreamContext const&, ModuleCallingContext const&);
      void postModule(StreamContext const&, ModuleCallingContext const&);
      
      double curr_job_time_;    // seconds
      double curr_job_cpu_;     // seconds
                                //use last run time for determining end of processing
      std::atomic<double> last_run_time_;
      std::atomic<double> last_run_cpu_;
      std::vector<double> curr_events_time_;  // seconds
      bool summary_only_;
      bool report_summary_;
      
      //
      // Min Max and total event times for each Stream.
      //  Used for summary at end of job
      std::vector<double> max_events_time_; // seconds
      std::vector<double> min_events_time_; // seconds
      std::vector<double> sum_events_time_;
      std::atomic<unsigned long> total_event_count_;
      unsigned int nStreams_;
      unsigned int nThreads_;
    };
  }
}

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
      if(gettimeofday(&t, 0) < 0)
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
    static
    std::vector<double>& moduleTimeStack() {
      static thread_local std::vector<double> s_stack;
      return s_stack;
    }

    Timing::Timing(ParameterSet const& iPS, ActivityRegistry& iRegistry) :
        curr_job_time_(0.),
        curr_job_cpu_(0.),
        last_run_time_(0.0),
        last_run_cpu_(0.0),
        curr_events_time_(),
        summary_only_(iPS.getUntrackedParameter<bool>("summaryOnly")),
        report_summary_(iPS.getUntrackedParameter<bool>("useJobReport")),
        max_events_time_(),
        min_events_time_(),
        total_event_count_(0) {
      iRegistry.watchPostBeginJob(this, &Timing::postBeginJob);
      iRegistry.watchPostEndJob(this, &Timing::postEndJob);

      iRegistry.watchPreEvent(this, &Timing::preEvent);
      iRegistry.watchPostEvent(this, &Timing::postEvent);

      if(not summary_only_) {
        iRegistry.watchPreModuleEvent(this, &Timing::preModule);
        iRegistry.watchPostModuleEvent(this, &Timing::postModule);
      }
          
      iRegistry.preallocateSignal_.connect([this](service::SystemBounds const& iBounds){
        nStreams_ = iBounds.maxNumberOfStreams();
        nThreads_ = iBounds.maxNumberOfThreads();
        curr_events_time_.resize(nStreams_,0.);
        sum_events_time_.resize(nStreams_,0.);
        max_events_time_.resize(nStreams_,0.);
        min_events_time_.resize(nStreams_,1.E6);
        
      });
      
      iRegistry.postGlobalEndRunSignal_.connect([this](edm::GlobalContext const&) {
        last_run_time_ = getTime();
        last_run_cpu_ = getCPU();
      });
    }

    Timing::~Timing() {
    }
    
    void Timing::addToCPUTime(StreamID id, double iTime) {
      //For accounting purposes we effectively can say we started earlier
      curr_job_cpu_ -= iTime;
    }

    double Timing::getTotalCPU() const {
      return getCPU();
    }

    void Timing::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.addUntracked<bool>("summaryOnly", false)->setComment(
      "If 'true' do not report timing for each event");
      desc.addUntracked<bool>("useJobReport", true)->setComment(
       "If 'true' write summary information to JobReport");
      descriptions.add("Timing", desc);
      descriptions.setComment(
       "This service reports the time it takes to run each module in a job.");
    }

    void Timing::postBeginJob() {
      curr_job_time_ = getTime();
      curr_job_cpu_ = getCPU();

      if(not summary_only_) {
        LogImportant("TimeReport")
        << "TimeReport> Report activated" << "\n"
        << "TimeReport> Report columns headings for events: "
        << "eventnum runnum timetaken\n"
        << "TimeReport> Report columns headings for modules: "
        << "eventnum runnum modulelabel modulename timetakeni\n"
        << "TimeReport> JobTime=" << curr_job_time_  << " JobCPU=" << curr_job_cpu_  << "\n";
      }
    }

    void Timing::postEndJob() {
      double total_job_time = getTime() - curr_job_time_;

      double total_job_cpu = getCPU() - curr_job_cpu_;

      double min_event_time = *(std::min_element(min_events_time_.begin(),
                                                 min_events_time_.end()));
      double max_event_time = *(std::max_element(max_events_time_.begin(),
                                               max_events_time_.end()));

      auto total_loop_time = last_run_time_ - curr_job_time_;
      auto total_loop_cpu = last_run_cpu_ - curr_job_cpu_;

      if(last_run_time_ == 0.0) {
        total_loop_time = 0.0;
        total_loop_cpu = 0.0;
      }

      double sum_all_events_time = 0;
      for(auto t : sum_events_time_) { sum_all_events_time += t; }

      double average_event_time = 0.0;
      if(total_event_count_ != 0) {
        average_event_time = sum_all_events_time / total_event_count_;
      }

      double event_throughput = 0.0;
      if(total_loop_time != 0.0) {
        event_throughput = total_event_count_ / total_loop_time;
      }

      LogImportant("TimeReport")
        << "TimeReport> Time report complete in "
        << total_job_time << " seconds"
        << "\n"
        << " Time Summary: \n"
        << " - Min event:   " << min_event_time << "\n"
        << " - Max event:   " << max_event_time << "\n"
        << " - Avg event:   " << average_event_time << "\n"
        << " - Total loop:  " <<total_loop_time <<"\n"
        << " - Total job:   " << total_job_time << "\n"
        << " Event Throughput: "<< event_throughput <<" ev/s\n"
        << " CPU Summary: \n"
        << " - Total loop:  " << total_loop_cpu << "\n"
        << " - Total job:   " << total_job_cpu << "\n";

      if(report_summary_) {
        Service<JobReport> reportSvc;
        std::map<std::string, std::string> reportData;

        reportData.insert(std::make_pair("MinEventTime", d2str(min_event_time)));
        reportData.insert(std::make_pair("MaxEventTime", d2str(max_event_time)));
        reportData.insert(std::make_pair("AvgEventTime", d2str(average_event_time)));
        reportData.insert(std::make_pair("EventThroughput", d2str(event_throughput)));
        reportData.insert(std::make_pair("TotalJobTime", d2str(total_job_time)));
        reportData.insert(std::make_pair("TotalJobCPU", d2str(total_job_cpu)));
        reportData.insert(std::make_pair("TotalLoopCPU", d2str(total_loop_cpu)));
        reportData.insert(std::make_pair("NumberOfStreams",ui2str(nStreams_)));
        reportData.insert(std::make_pair("NumberOfThreads",ui2str(nThreads_)));
        reportSvc->reportPerformanceSummary("Timing", reportData);
      }
    }

    void Timing::preEvent(StreamContext const& iStream) {
      auto index = iStream.streamID().value();
      curr_events_time_[index] = getTime();
    }

    void Timing::postEvent(StreamContext const& iStream) {
      auto index = iStream.streamID().value();

      double curr_event_time = getTime() - curr_events_time_[index];
      sum_events_time_[index] +=curr_event_time;

      if(not summary_only_) {
        auto const & eventID = iStream.eventID();
        LogPrint("TimeEvent")
        << "TimeEvent> "
        << eventID.event() << " "
        << eventID.run() << " "
        << curr_event_time ;
      }
      if(curr_event_time > max_events_time_[index]) max_events_time_[index] = curr_event_time;
      if(curr_event_time < min_events_time_[index]) min_events_time_[index] = curr_event_time;
      ++total_event_count_;
    }

    void Timing::preModule(StreamContext const&, ModuleCallingContext const&) {
      auto & modStack = moduleTimeStack();
      modStack.push_back(getTime());
    }

    void Timing::postModule(StreamContext const& iStream, ModuleCallingContext const& iModule) {
      //LogInfo("TimeModule")
      auto& modStack = moduleTimeStack();
      assert(modStack.size() > 0);
      double curr_module_time = modStack.back();
      modStack.pop_back();
      double t = getTime() - curr_module_time;
      //move waiting module start times forward to account
      // for the fact that they were paused while this module ran
      for(auto& waitingModuleStartTime : modStack) {
        waitingModuleStartTime +=t;
      }
      auto const & eventID = iStream.eventID();
      auto const & desc = *(iModule.moduleDescription());
      
      LogPrint("TimeModule") << "TimeModule> "
      << eventID.event() << " "
      << eventID.run() << " "
      << desc.moduleLabel() << " "
      << desc.moduleName() << " "
      << t;
    }
  }
}

using edm::service::Timing;

typedef edm::serviceregistry::AllArgsMaker<edm::TimingServiceBase,Timing> TimingMaker;
DEFINE_FWK_SERVICE_MAKER(Timing, TimingMaker);
