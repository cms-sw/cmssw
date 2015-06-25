#ifndef DQMOFFLINE_TRIGGER_BTVHLTOFFLINECLIENT
#define DQMOFFLINE_TRIGGER_BTVHLTOFFLINECLIENT

// -*- C++ -*-
//
// Package:    BTVHLTOffline
// Class:      BTVHLTOffline
// 
/*
 Class does nothing so far..
 */
//
// Originally created by:  Anne-Catherine Le Bihan
//                         June 2015
// Following the structure used in JetMetHLTOfflineClient

#include <memory>
#include <unistd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//#include "DataFormats/JetReco/interface/CaloJetCollection.h"
//#include "DataFormats/METReco/interface/CaloMETCollection.h"
//#include "DataFormats/METReco/interface/CaloMET.h"

#include <iostream>
#include <fstream>
#include <vector>

class DQMStore;
class MonitorElement;

class BTVHLTOfflineClient : public DQMEDHarvester {
 
 private:
  edm::ParameterSet conf_;

  bool debug_;
  bool verbose_;

  std::string dirName_;
  std::string hltTag_;
  std::string processname_;

 public:
  explicit BTVHLTOfflineClient(const edm::ParameterSet& );
  virtual ~BTVHLTOfflineClient();

  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob
  
};
 
#endif
