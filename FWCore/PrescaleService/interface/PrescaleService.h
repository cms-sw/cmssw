// -*- C++ -*-
//
// Package:     PrescaleService
// Class  :     PrescaleService
//
// Implementation:
//     Cache and make prescale factors available online.
//
// Current revision: $Revision: 1.1 $
// On branch: $Name: V00-00-00 $
// Latest change by $Author: wmtan $ at $Date: 2007/04/23 23:45:42 $
//


#ifndef FWCore_PrescaleService_PrescaleService_h
#define FWCore_PrescaleService_PrescaleService_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Provenance/interface/EventID.h"

#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Framework/interface/TriggerReport.h"

#include "boost/thread/mutex.hpp"

using namespace std;

#include <vector>
#include <iostream>


namespace edm {
  namespace service {

    class PrescaleService
    {
	
    private:

      boost::mutex mutex;        // protect vectors
      edm::EventID curr_event_;
      int count_;                // counter incremented in postEventProcessing
      edm::EventProcessor *fu_;  // pointer to FUEP
      edm::TriggerReport tr_;    // trigger report
      edm::TriggerReport trold;  // trigger report at start of event
      unsigned int lsold;        // current LS block number

      vector<string> prescalers; // prescaler cache
      vector<string> triggers;   // trigger counter cache
      string stsstr;             // last status string sent
      string trgstr;             // last trigger statistics string sent

      unsigned int blsn;         // putPrescaler error decoding LS#
      unsigned int bpath;        // putPrescaler error decoding path
      unsigned int bmod;         // putPrescaler error decoding module
      unsigned int bfac;         // putPrescaler error decoding factor
      unsigned int berr;         // putPrescaler no path/module/factors found
      unsigned int lsgmax;       // getPrescaler max LS#
      unsigned int glow;         // getPrescaler LS < lsgmax, i.e. mixed order event
      unsigned int lspmax;       // putPrescaler max LS#
      unsigned int pleq;         // putPrescaler LS <= lspmax, i.e. mixed order prescaler update

      unsigned int lsg;          // getPrescaler max LS# associated with cache
      unsigned int lsc;          // cached LS# associated with lsg
      unsigned int bang;         // error count of spoilt events


    public:

      PrescaleService(const ParameterSet&,ActivityRegistry&);
      ~PrescaleService();

      void postBeginJob();
      void postEndJob();
      
      void preEventProcessing(const edm::EventID&, const edm::Timestamp&);
      void postEventProcessing(const edm::Event&, const edm::EventSetup&);
      
      void preModule(const ModuleDescription&);
      void postModule(const ModuleDescription&);

      int getPrescale(unsigned int ls, string module);
      int putPrescale(string s);
      int sizePrescale();
      void putHandle(edm::EventProcessor *proc_);

      string getStatus();
      string getLs();
      string getLs(string lsAsString);
	
    };
  }
}

#endif
