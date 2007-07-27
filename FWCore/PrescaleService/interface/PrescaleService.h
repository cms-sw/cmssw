#ifndef FWCore_PrescaleService_PrescaleService_h
#define FWCore_PrescaleService_PrescaleService_h

// -*- C++ -*-
//
// Package:     PrescaleService
// Class  :     PrescaleService
//
// Implementation:
//     Cache and make prescale factors available online.
//
// Current revision: $Revision: 1.3 $
// On branch: $Name:  $
// Latest change by $Author: wmtan $ at $Date: 2007/06/14 21:03:39 $
//

#include "DataFormats/Provenance/interface/EventID.h"

#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Framework/interface/TriggerReport.h"

#include "boost/thread/mutex.hpp"

#include <string>
#include <vector>

namespace edm {
  namespace service {

    class PrescaleService
    {
	
    private:

      boost::mutex mutex;        // protect std::vectors
      edm::EventID curr_event_;
      unsigned int count_;       // counter incremented in postEventProcessing
      edm::EventProcessor *fu_;  // pointer to FUEP
      edm::TriggerReport tr_;    // trigger report
      edm::TriggerReport trold;  // trigger report at start of event
      unsigned int lsold;        // current LS block number

      std::vector<std::string> prescalers; // prescaler cache
      std::vector<std::string> triggers;   // trigger counter cache
      std::string stsstr;             // last status std::string sent
      std::string trgstr;             // last trigger statistics std::string sent
      std::string trstr;              // trigger report string

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
      unsigned int nops;         // count of empty prescaler vector

    public:

      PrescaleService(const ParameterSet&,ActivityRegistry&);
      ~PrescaleService();

      void postBeginJob();
      void postEndJob();
      
      void preEventProcessing(const edm::EventID&, const edm::Timestamp&);
      void postEventProcessing(const edm::Event&, const edm::EventSetup&);
      
      void preModule(const ModuleDescription&);
      void postModule(const ModuleDescription&);

      int getPrescale(unsigned int ls, std::string module);
      int getPrescale(std::string module);
      int putPrescale(std::string s);
      int sizePrescale();
      void putHandle(edm::EventProcessor *proc_);

      std::string getStatus();
      std::string getLs();
      std::string getLs(std::string lsAsString);
      std::string getTr();
	
    };
  }
}

#endif

