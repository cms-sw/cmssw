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
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/MonitorElement.h"

class DQMStore;

class MonitorLTC : public edm::EDAnalyzer {
   public:
      explicit MonitorLTC(const edm::ParameterSet&);
      ~MonitorLTC(){};
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
       virtual void beginJob() ;
       virtual void endJob() ;
   private:
       DQMStore* dqmStore_;
       edm::ParameterSet conf_;
       // trigger decision from LTC digis
       MonitorElement * LTCTriggerDecision_all;
       //
       std::string HLTDirectory;
       edm::InputTag ltcDigiCollectionTag_;
};

#endif
