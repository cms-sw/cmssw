// -*- C++ -*-
//
// Package:     Services
// Class  :     Timing
// 
// Implementation:
//
// Original Author:  Jim Kowalkowski
// $Id: Timing.cc,v 1.1 2005/12/20 19:37:33 jbk Exp $
//

#include "FWCore/Services/src/Timing.h"
#include "FWCore/Framework/interface/ModuleDescription.h"
#include "DataFormats/Common/interface/EventID.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <sys/time.h>

using namespace std;

namespace edm {
  namespace service {

    static double getTime()
    {
      struct timeval t;
      if(gettimeofday(&t,0)<0)
	throw cms::Exception("SysCallFailed","Failed call to gettimeofday");

      return (double)t.tv_sec + (double(t.tv_usec) * 1E-6);
    }

    Timing::Timing(const ParameterSet& iPS, ActivityRegistry&iRegistry):
      want_summary_(iPS.getUntrackedParameter<bool>("summaryOnly",false))
    {
      iRegistry.watchPostBeginJob(this,&Timing::postBeginJob);
      iRegistry.watchPostEndJob(this,&Timing::postEndJob);

      iRegistry.watchPreProcessEvent(this,&Timing::preEventProcessing);
      iRegistry.watchPostProcessEvent(this,&Timing::postEventProcessing);

      iRegistry.watchPreModule(this,&Timing::preModule);
      iRegistry.watchPostModule(this,&Timing::postModule);
    }


    Timing::~Timing()
    {
    }

    void Timing::postBeginJob()
    {
      // edm::LogInfo("TimeReport")
      cout
	<< "TimeReport> Report activated" << "\n"
	<< "TimeReport> Report columns headings for events: "
	<< "eventnum runnum timetaken\n"
	<< "TimeReport> Report columns headings for modules: "
	<< "eventnum runnum modulelabel modulename timetaken\n"
	<< "\n";

      curr_job_ = getTime();
    }

    void Timing::postEndJob()
    {
      double t = getTime() - curr_job_;
      // edm::LogInfo("TimeReport")
      cout
	<< "TimeReport> Time report complete in "
	<< t << " seconds"
	<< "\n";
    }

    void Timing::preEventProcessing(const edm::EventID& iID,
				    const edm::Timestamp& iTime)
    {
      curr_event_ = iID;
      curr_event_time_ = getTime();
    }
    void Timing::postEventProcessing(const Event& e, const EventSetup&)
    {
      double t = getTime() - curr_event_time_;
      //edm::LogInfo("TimeEvent")
      cout << "TimeEvent> "
	   << curr_event_.event() << " "
	   << curr_event_.run() << " "
	   << t << "\n";
    }

    void Timing::preModule(const ModuleDescription&)
    {
      curr_module_time_ = getTime();
    }

    void Timing::postModule(const ModuleDescription& desc)
    {
      double t = getTime() - curr_module_time_;
      //edm::LogInfo("TimeModule")
      cout << "TimeModule> "
	   << curr_event_.event() << " "
	   << curr_event_.run() << " "
	   << desc.moduleLabel_ << " "
	   << desc.moduleName_ << " "
	   << t << "\n";
    }

  }
}
