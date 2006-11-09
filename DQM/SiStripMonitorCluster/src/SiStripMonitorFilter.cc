// -*- C++ -*-
//
// Package:    SiStripMonitorCluster
// Class:      SiStripMonitorFilter
// 
//class SiStripMonitorFilter SiStripMonitorFilter.cc DQM/SiStripMonitorCluster/src/SiStripMonitorFilter.cc
#include <vector>

#include <numeric>
#include <iostream>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQM/SiStripMonitorCluster/interface/SiStripMonitorFilter.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

using namespace edm;

SiStripMonitorFilter::SiStripMonitorFilter(const edm::ParameterSet& iConfig)
{
  FilterDirectory="FilterResults";
  dbe_  = Service<DaqMonitorBEInterface>().operator->();
  conf_ = iConfig;
}

void SiStripMonitorFilter::beginJob(const edm::EventSetup& es){
  dbe_->setCurrentFolder(FilterDirectory);
  std::string FilterProducer = conf_.getParameter<std::string>("FilterProducer");
  FilterDecision = dbe_->book1D(FilterProducer+"_Decision", FilterProducer+"Decision", 2, -0.5, 1.5);
}

void SiStripMonitorFilter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // get from event
  std::string FilterProducer = conf_.getParameter<std::string>("FilterProducer");
  Handle<int> filter_decision; iEvent.getByLabel(FilterProducer, "", filter_decision); // filter decision
  // trigger decision
  FilterDecision->Fill(*filter_decision);
}

void SiStripMonitorFilter::endJob(void){
  double events_accepted = FilterDecision->getBinContent(1);
  double events_rejected = FilterDecision->getBinContent(2);
  double events_total    = events_accepted + events_rejected;
  LogInfo("DQM|SiStripMonitorFilter")<<"Total nr. of events "<<events_total;
  LogInfo("DQM|SiStripMonitorFilter")<<"Events rejected/accepted "<<events_accepted<<"/"<<events_rejected;
  LogInfo("DQM|SiStripMonitorFilter")<<"rejected/total  :  accepted/total "<<events_rejected/events_total<<"  :  "<<events_accepted/events_total;
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dbe_->save(outputFileName);
  }
}

