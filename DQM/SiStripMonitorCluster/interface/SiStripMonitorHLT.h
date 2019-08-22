#ifndef SiStripMonitorCluster_SiStripMonitorHLT_h
#define SiStripMonitorCluster_SiStripMonitorHLT_h
// -*- C++ -*-
//
// Package:     SiStripMonitorCluster
// Class  :     SiStripMonitorHLT

// system include files
#include <memory>

// user include files
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

class SiStripMonitorHLT : public DQMEDAnalyzer {
public:
  explicit SiStripMonitorHLT(const edm::ParameterSet &);
  ~SiStripMonitorHLT() override{};

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  edm::EDGetTokenT<int> filerDecisionToken_;
  edm::EDGetTokenT<uint> sumOfClusterToken_;
  edm::EDGetTokenT<std::map<uint, std::vector<SiStripCluster> > > clusterInSubComponentsToken_;

  edm::ParameterSet conf_;
  MonitorElement *HLTDecision;
  // all events
  MonitorElement *SumOfClusterCharges_all;
  MonitorElement *NumberOfClustersAboveThreshold_all;
  MonitorElement *ChargeOfEachClusterTIB_all;
  MonitorElement *ChargeOfEachClusterTOB_all;
  MonitorElement *ChargeOfEachClusterTEC_all;
  // events that passes the HLT
  MonitorElement *SumOfClusterCharges_hlt;
  MonitorElement *NumberOfClustersAboveThreshold_hlt;
  MonitorElement *ChargeOfEachClusterTIB_hlt;
  MonitorElement *ChargeOfEachClusterTOB_hlt;
  MonitorElement *ChargeOfEachClusterTEC_hlt;
  //
  std::string HLTDirectory;
};

#endif
