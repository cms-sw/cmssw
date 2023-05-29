#ifndef DQMOFFLINE_TRIGGER_JETMETHLTOFFLINECLIENT
#define DQMOFFLINE_TRIGGER_JETMETHLTOFFLINECLIENT

// -*- C++ -*-
//
// Package:    JetMETHLTOffline
// Class:      JetMETHLTOffline
//
/*
 Description: This is a DQM client meant to plot high-level HLT trigger quantities 
 as stored in the HLT results object TriggerResults for the JetMET triggers
*/

//
// Originally create by:  Kenichi Hatakeyama
//                        April 2009
//
// Migrated to use DQMEDHarvester by: Jyothsna Rani Komaragiri, Oct 2014
//

#include <memory>
#include <unistd.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"

#include <iostream>
#include <fstream>
#include <vector>

class JetMETHLTOfflineClient : public DQMEDHarvester {
private:
  edm::ParameterSet conf_;

  bool debug_;
  bool verbose_;

  std::string dirName_;
  std::string hltTag_;
  std::string processname_;

public:
  explicit JetMETHLTOfflineClient(const edm::ParameterSet &);
  ~JetMETHLTOfflineClient() override;

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;  //performed in the endJob
};

#endif
