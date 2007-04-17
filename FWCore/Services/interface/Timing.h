#ifndef Services_TIMING_h
#define Services_TIMING_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     Timing
// 
//
// Original Author:  Jim Kowalkowski
// $Id: Timing.h,v 1.3 2006/08/16 13:36:28 chrjones Exp $
//
#include "sigc++/signal.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Common/interface/EventID.h"

namespace edm {
  namespace service {
    class Timing
    {
    public:
      Timing(const ParameterSet&,ActivityRegistry&);
      ~Timing();

      sigc::signal<void, const edm::ModuleDescription&, double> newMeasurementSignal;
    private:
      void postBeginJob();
      void postEndJob();
      
      void preEventProcessing(const edm::EventID&, const edm::Timestamp&);
      void postEventProcessing(const Event&, const EventSetup&);
      
      void preModule(const ModuleDescription&);
      void postModule(const ModuleDescription&);

      edm::EventID curr_event_;
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
