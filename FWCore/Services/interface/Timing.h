#ifndef Services_TIMING_h
#define Services_TIMING_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     Timing
// 
//
// Original Author:  Jim Kowalkowski
// $Id: Timing.h,v 1.5 2006/12/20 00:22:46 wmtan Exp $
//
#include "sigc++/signal.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"

namespace edm {
  struct ActivityRegistry;
  class Event;
  class EventSetup;
  class ParameterSet;
  namespace service {
    class Timing
    {
    public:
      Timing(const ParameterSet&,ActivityRegistry&);
      ~Timing();

      sigc::signal<void, const ModuleDescription&, double> newMeasurementSignal;
    private:
      void postBeginJob();
      void postEndJob();
      
      void preEventProcessing(const EventID&, const Timestamp&);
      void postEventProcessing(const Event&, const EventSetup&);
      
      void preModule(const ModuleDescription&);
      void postModule(const ModuleDescription&);

      EventID curr_event_;
      double curr_job_; // seconds
      double curr_event_time_;  // seconds
      double curr_module_time_; // seconds
      bool summary_only_;
      bool report_summary_;
      
        //
       // Min Max and average event times for summary
      //  at end of job
      double max_event_time_;    // seconds
      double min_event_time_;    // seconds
      int total_event_count_; 
    };
  }
}



#endif
