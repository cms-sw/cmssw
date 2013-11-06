#ifndef Services_TIMING_h
#define Services_TIMING_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     Timing
//
// Original Author:  Jim Kowalkowski
//

#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include <atomic>

namespace edm {
  class ActivityRegistry;
  class Event;
  class EventSetup;
  class ParameterSet;
  class ConfigurationDescriptions;
  class StreamContext;
  class ModuleCallingContext;

  namespace service {
    class Timing {
    public:
      Timing(ParameterSet const&,ActivityRegistry&);
      ~Timing();

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    private:
      
      void postBeginJob();
      void postEndJob();

      void preEvent(StreamContext const&);
      void postEvent(StreamContext const&);

      void preModule(StreamContext const&, ModuleCallingContext const&);
      void postModule(StreamContext const&, ModuleCallingContext const&);

      double curr_job_time_;    // seconds
      double curr_job_cpu_;     // seconds
      std::vector<double> curr_events_time_;  // seconds
      std::vector<double> curr_events_cpu_;   // seconds
      std::vector<double> total_events_cpu_;  // seconds
      bool summary_only_;
      bool report_summary_;

      //
      // Min Max and average event times for each Stream.
      //  Used for summary at end of job
      std::vector<double> max_events_time_; // seconds
      std::vector<double> max_events_cpu_;  // seconds
      std::vector<double> min_events_time_; // seconds
      std::vector<double> min_events_cpu_;  // seconds
      std::atomic<unsigned long> total_event_count_;
    };
  }
}

#endif
