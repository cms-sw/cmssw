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

void SiStripMonitorFilter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<int> filter_decision; iEvent.getByToken(filerDecisionToken_,filter_decision); // filter decision

  // trigger decision
  FilterDecision->Fill(*filter_decision);
}
