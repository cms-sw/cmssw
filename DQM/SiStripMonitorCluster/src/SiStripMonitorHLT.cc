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
  HLTDecision = dbe_->book1D("HLTDecision", "HLTDecision", 2, -0.5, 1.5);
//  HLTDecision->GetXaxis()->SetBinLabel(1,"Rejected");
//  HLTDecision->GetXaxis()->SetBinLabel(2,"Accepted");
  ClusterCharge = dbe_->book1D("Cluster Charge", "Cluster Charge", 50, 0, 1500);
}


void SiStripMonitorHLT::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  std::string HLTProducer = conf_.getParameter<std::string>("HLTProducer");

   Handle<uint> clustch;
   iEvent.getByLabel(HLTProducer, "", clustch);

   Handle<int> hltres;
   iEvent.getByLabel(HLTProducer, "", hltres);


   ClusterCharge->Fill(*clustch);
   std::cout<<"ClusterCharge :"<<*clustch<<std::endl;
   HLTDecision->Fill(*hltres);
}


void SiStripMonitorHLT::endJob(void){
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  string outputFileName = conf_.getParameter<string>("OutputFileName");
  if(outputMEsInRootFile){
    dbe_->save(outputFileName);
  }
}


