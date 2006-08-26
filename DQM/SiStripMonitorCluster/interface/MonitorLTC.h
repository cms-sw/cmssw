#ifndef SiStripMonitorCluster_MonitorLTC_h
#define SiStripMonitorCluster_MonitorLTC_h
// -*- C++ -*-
//
// Package:     SiStripMonitorCluster
// Class  :     MonitorLTC



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElement.h"

class MonitorLTC : public edm::EDAnalyzer {
   public:
      explicit MonitorLTC(const edm::ParameterSet&);
      ~MonitorLTC(){};
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
       virtual void beginJob(edm::EventSetup const&) ;
       virtual void endJob() ;
   private:
       DaqMonitorBEInterface* dbe_;
       edm::ParameterSet conf_;
       // trigger decision from LTC digis
       MonitorElement * LTCTriggerDecision_all;
       //
       std::string HLTDirectory;
};

#endif
