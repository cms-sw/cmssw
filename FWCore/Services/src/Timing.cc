// -*- C++ -*-
//
// Package:     Services
// Class  :     Timing
//
// Implementation:
//
// Original Author:  Jim Kowalkowski
//
// Change Log
//
// 1 - mf 4/22/08   Facilitate summary output to job report and logs:
//		    In Timing ctor, default for report_summary_ changed to true
//                  In postEndJob, add output to logger
//
// 2 - 2009/01/14 10:29:00, Natalia Garcia Nebot
//        Modified the service to add some new measurements to report:
//                - Average time per event (cpu and wallclock)
//                - Fastest time per event (cpu and wallclock)
//                - Slowest time per event (cpu and wallclock)
//
// 3 - mf 3/18/09  Change use of LogAbsolute to LogImportant
//		   so that users can throttle the messages
//                 for selected destinations.  LogImportant
//                 is treated at the same level as LogError, so
//                 by default the behavior will not change, but
//                 there will now be a way to control the verbosity.
//
// 4 - mf 3/18/09  The per-event output TimeEvent is changed to LogPrint.
//		   The per-module output TimeModule is changed to LogPrint.
//
// 5 - wmtan 2010/03/16
//		   Fixed the constructor to initialize all data members!
//		   Standardized coding style
//

#include "FWCore/Services/interface/Timing.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <iostream>
#include <sys/time.h>
#include <sys/resource.h>
#include <sstream>

namespace edm {
  namespace service {

    static std::string d2str(double d) {
	std::stringstream t;
	t << d;
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
	return totalCPUTime;
    }

    Timing::Timing(ParameterSet const& iPS, ActivityRegistry& iRegistry) :
        curr_event_(),
        curr_job_time_(0.),
        curr_job_cpu_(0.),
        curr_event_time_(0.),
        curr_event_cpu_(0.),
        curr_module_time_(0.),
        total_event_cpu_(0.),
        summary_only_(iPS.getUntrackedParameter<bool>("summaryOnly")),
        report_summary_(iPS.getUntrackedParameter<bool>("useJobReport")),
        max_event_time_(0.),
        max_event_cpu_(0.),
        min_event_time_(0.),
        min_event_cpu_(0.),
        total_event_count_(0) {
      iRegistry.watchPostBeginJob(this, &Timing::postBeginJob);
      iRegistry.watchPostEndJob(this, &Timing::postEndJob);

      iRegistry.watchPreProcessEvent(this, &Timing::preEventProcessing);
      iRegistry.watchPostProcessEvent(this, &Timing::postEventProcessing);

      iRegistry.watchPreModule(this, &Timing::preModule);
      iRegistry.watchPostModule(this, &Timing::postModule);
    }

    Timing::~Timing() {
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
      total_event_cpu_ = 0.0;

      //LogInfo("TimeReport")
      if (not summary_only_) {
        LogImportant("TimeReport")				// ChangeLog 3
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
      double average_event_time = total_job_time / total_event_count_;

      double total_job_cpu = getCPU() - curr_job_cpu_;
      double average_event_cpu = total_event_cpu_ / total_event_count_;

      //LogAbsolute("FwkJob")
      //LogAbsolute("TimeReport")				// Changelog 1
      LogImportant("TimeReport")				// Changelog 3
	<< "TimeReport> Time report complete in "
	<< total_job_time << " seconds"
	<< "\n"
        << " Time Summary: \n"
        << " - Min event:   " << min_event_time_ << "\n"
        << " - Max event:   " << max_event_time_ << "\n"
        << " - Avg event:   " << average_event_time << "\n"
	<< " - Total job:   " << total_job_time << "\n"
	<< " CPU Summary: \n"
        << " - Min event:   " << min_event_cpu_ << "\n"
        << " - Max event:   " << max_event_cpu_ << "\n"
        << " - Avg event:   " << average_event_cpu << "\n"
        << " - Total job:   " << total_job_cpu << "\n"
        << " - Total event: " << total_event_cpu_ << "\n";

      if (report_summary_) {
	Service<JobReport> reportSvc;
//	std::map<std::string, double> reportData;
	std::map<std::string, std::string> reportData;

	reportData.insert(std::make_pair("MinEventTime", d2str(min_event_time_)));
	reportData.insert(std::make_pair("MaxEventTime", d2str(max_event_time_)));
	reportData.insert(std::make_pair("AvgEventTime", d2str(average_event_time)));
	reportData.insert(std::make_pair("TotalJobTime", d2str(total_job_time)));
        reportData.insert(std::make_pair("MinEventCPU", d2str(min_event_cpu_)));
        reportData.insert(std::make_pair("MaxEventCPU", d2str(max_event_cpu_)));
        reportData.insert(std::make_pair("AvgEventCPU", d2str(average_event_cpu)));
	reportData.insert(std::make_pair("TotalJobCPU", d2str(total_job_cpu)));
        reportData.insert(std::make_pair("TotalEventCPU", d2str(total_event_cpu_)));
	
	reportSvc->reportPerformanceSummary("Timing", reportData);
//	reportSvc->reportTimingInfo(reportData);
      }
    }

    void Timing::preEventProcessing(EventID const& iID, Timestamp const& iTime) {
      curr_event_ = iID;
      curr_event_time_ = getTime();
      curr_event_cpu_ = getCPU();
    }

    void Timing::postEventProcessing(Event const& e, EventSetup const&) {
      curr_event_cpu_ = getCPU() - curr_event_cpu_;
      total_event_cpu_ += curr_event_cpu_;

      curr_event_time_ = getTime() - curr_event_time_;

      if (not summary_only_) {
        LogPrint("TimeEvent")				// ChangeLog 3
	<< "TimeEvent> "
	<< curr_event_.event() << " "
	<< curr_event_.run() << " "
	<< curr_event_time_ << " "
	<< curr_event_cpu_ << " "
	<< total_event_cpu_;
      }
      if (total_event_count_ == 0) {
	max_event_time_ = curr_event_time_;
        min_event_time_ = curr_event_time_;
	max_event_cpu_ = curr_event_cpu_;
        min_event_cpu_ = curr_event_cpu_;
      }

      if (curr_event_time_ > max_event_time_) max_event_time_ = curr_event_time_;
      if (curr_event_time_ < min_event_time_) min_event_time_ = curr_event_time_;
      if (curr_event_cpu_ > max_event_cpu_) max_event_cpu_ = curr_event_cpu_;
      if (curr_event_cpu_ < min_event_cpu_) min_event_cpu_ = curr_event_cpu_;
      total_event_count_ = total_event_count_ + 1;
    }

    void Timing::preModule(ModuleDescription const&) {
      curr_module_time_ = getTime();
    }

    void Timing::postModule(ModuleDescription const& desc) {
      double t = getTime() - curr_module_time_;
      //LogInfo("TimeModule")
      if (not summary_only_) {
        LogPrint("TimeModule") << "TimeModule> "		// ChangeLog 4
	   << curr_event_.event() << " "
	   << curr_event_.run() << " "
	   << desc.moduleLabel() << " "
	   << desc.moduleName() << " "
	   << t;
      }

      newMeasurementSignal(desc, t);
    }
  }
}

