// -*- C++ -*-
//
// Package:    SiStripMonitorCluster
// Class:      SiStripMonitorHLT
// 
//class SiStripMonitorHLT SiStripMonitorHLT.cc DQM/SiStripMonitorCluster/src/SiStripMonitorHLT.cc
#include <vector>

#include <numeric>
#include <iostream>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/SiStripMonitorCluster/interface/SiStripMonitorHLT.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

using namespace std;
using namespace edm;

SiStripMonitorHLT::SiStripMonitorHLT(const edm::ParameterSet& iConfig)
{
  dbe_  = edm::Service<DaqMonitorBEInterface>().operator->();
  conf_ = iConfig;
}


void SiStripMonitorHLT::beginJob(const edm::EventSetup& es){
  using namespace edm;

  dbe_->setCurrentFolder("HLTResults");
  std::string HLTProducer = conf_.getParameter<std::string>("HLTProducer");
  HLTDecision = dbe_->book1D(HLTProducer+"_HLTDecision", HLTProducer+"HLTDecision", 2, -0.5, 1.5);
  ClusterCharge = dbe_->book1D("SumOfClusterCharges", "SumOfClusterCharges", 50, 0, 1500);
  // 1 = TIB2, 2 = TIB2, 3 = TIB3, 4 = TOB, 5 = TEC
  NumberOfClustersAboveThreshold = dbe_->book1D("NumberOfClustersAboveThreshold", "NumberOfClustersAboveThreshold", 30, 30.5, 60.5);
}

void SiStripMonitorHLT::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  std::string HLTProducer = conf_.getParameter<std::string>("HLTProducer");

   // sum of cluster charges
   Handle<uint> sum_of_clustch; iEvent.getByLabel(HLTProducer, "", sum_of_clustch);
   // first element of pair: layer: TIB1, ...., TEC; second element: nr of clusters above threshold
   Handle<std::pair<uint,uint> > layers_nrclusters; 
//   Handle< std::map< uint, std::pair<SiStripCluster,DetId> > > clusters_in_subcomponents;
//   if(HLTProducer=="ClusterMTCCFilter") iEvent.getByLabel(HLTProducer, "", clusters_in_subcomponents);

   // filter decision
   Handle<int> hltres; iEvent.getByLabel(HLTProducer, "", hltres);

//   if(HLTProducer=="ClusterMTCCFilter") NumberOfClustersAboveThreshold->Fill( (*layers_nrclusters).first, (*layers_nrclusters).second);
   ClusterCharge->Fill(*sum_of_clustch);
   HLTDecision->Fill(*hltres);
}

void SiStripMonitorHLT::endJob(void){
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  string outputFileName = conf_.getParameter<string>("OutputFileName");
  if(outputMEsInRootFile){
    dbe_->save(outputFileName);
  }
}


