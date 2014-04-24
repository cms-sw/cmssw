// -*- C++ -*-
//
// Package:    SiStripMonitorCluster
// Class:      SiStripMonitorFilter
// 
//class SiStripMonitorFilter SiStripMonitorFilter.cc DQM/SiStripMonitorCluster/src/SiStripMonitorFilter.cc
#include <vector>

#include <numeric>
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQM/SiStripMonitorCluster/interface/SiStripMonitorFilter.h"
#include "DQMServices/Core/interface/DQMStore.h"


SiStripMonitorFilter::SiStripMonitorFilter(const edm::ParameterSet& iConfig)
{
  FilterDirectory="FilterResults";
  dqmStore_  = edm::Service<DQMStore>().operator->();
  conf_ = iConfig;

  filerDecisionToken_ = consumes<int>(conf_.getParameter<std::string>("FilterProducer") );

}

void SiStripMonitorFilter::bookHistograms(DQMStore::IBooker & ibooker, const edm::Run & run, const edm::EventSetup & es) 
{
  ibooker.setCurrentFolder(FilterDirectory);
  std::string FilterProducer = conf_.getParameter<std::string>("FilterProducer");
  FilterDecision = ibooker.book1D(FilterProducer+"_Decision", FilterProducer+"Decision", 2, -0.5, 1.5);
  
}

void SiStripMonitorFilter::beginJob(){
}

void SiStripMonitorFilter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<int> filter_decision; iEvent.getByToken(filerDecisionToken_,filter_decision); // filter decision

  // trigger decision
  FilterDecision->Fill(*filter_decision);
}

void SiStripMonitorFilter::endJob(void){
  double events_accepted = FilterDecision->getBinContent(1);
  double events_rejected = FilterDecision->getBinContent(2);
  double events_total    = events_accepted + events_rejected;
  edm::LogInfo("DQM|SiStripMonitorFilter")<<"Total nr. of events "<<events_total;
  edm::LogInfo("DQM|SiStripMonitorFilter")<<"Events rejected/accepted "<<events_accepted<<"/"<<events_rejected;
  edm::LogInfo("DQM|SiStripMonitorFilter")<<"rejected/total  :  accepted/total "<<events_rejected/events_total<<"  :  "<<events_accepted/events_total;
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dqmStore_->save(outputFileName);
  }
}

