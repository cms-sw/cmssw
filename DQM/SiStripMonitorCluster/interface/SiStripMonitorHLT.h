#ifndef SiStripMonitorCluster_SiStripMonitorHLT_h
#define SiStripMonitorCluster_SiStripMonitorHLT_h
// -*- C++ -*-
//
// Package:     SiStripMonitorCluster
// Class  :     SiStripMonitorHLT



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElement.h"

class DQMStore;

class SiStripMonitorHLT : public edm::EDAnalyzer {
   public:
      explicit SiStripMonitorHLT(const edm::ParameterSet&);
      ~SiStripMonitorHLT(){};

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
       virtual void beginJob() ;
       virtual void endJob() ;

   private:
       DQMStore* dqmStore_;
       edm::ParameterSet conf_;
       MonitorElement * HLTDecision;
       // all events
       MonitorElement * SumOfClusterCharges_all;
       MonitorElement * NumberOfClustersAboveThreshold_all;
       MonitorElement * ChargeOfEachClusterTIB_all;
       MonitorElement * ChargeOfEachClusterTOB_all;
       MonitorElement * ChargeOfEachClusterTEC_all;
       // events that passes the HLT
       MonitorElement * SumOfClusterCharges_hlt;
       MonitorElement * NumberOfClustersAboveThreshold_hlt;
       MonitorElement * ChargeOfEachClusterTIB_hlt;
       MonitorElement * ChargeOfEachClusterTOB_hlt;
       MonitorElement * ChargeOfEachClusterTEC_hlt;
       //
       std::string HLTDirectory;
};

#endif
