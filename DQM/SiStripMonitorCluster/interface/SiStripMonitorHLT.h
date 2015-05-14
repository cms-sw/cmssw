#ifndef SiStripMonitorCluster_SiStripMonitorHLT_h
#define SiStripMonitorCluster_SiStripMonitorHLT_h
// -*- C++ -*-
//
// Package:     SiStripMonitorCluster
// Class  :     SiStripMonitorHLT



// system include files
#include <memory>

// user include files
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

class DQMStore;

class SiStripMonitorHLT : public DQMEDAnalyzer {
   public:
      explicit SiStripMonitorHLT(const edm::ParameterSet&);
      ~SiStripMonitorHLT(){};

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
       virtual void beginJob() ;
       virtual void endJob() ;
       void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

   private:

       edm::EDGetTokenT<int> filerDecisionToken_;
       edm::EDGetTokenT<uint> sumOfClusterToken_;
       edm::EDGetTokenT<std::map<uint,std::vector<SiStripCluster> > > clusterInSubComponentsToken_;

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
