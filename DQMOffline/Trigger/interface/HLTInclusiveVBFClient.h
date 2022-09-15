#ifndef DQMOFFLINE_TRIGGER_HLTInclusiveVBFClient
#define DQMOFFLINE_TRIGGER_HLTInclusiveVBFClient

// -*- C++ -*-
//
// Package:    HLTInclusiveVBFClient
// Class:      HLTInclusiveVBFClient
//

#include <memory>
#include <unistd.h>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
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

class HLTInclusiveVBFClient : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchLuminosityBlocks> {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;

private:
  DQMStore* dbe_;  //dbe seems to be the standard name for this, I dont know why. We of course dont own it

  edm::ParameterSet conf_;

  bool debug_;
  bool verbose_;

  std::string dirName_;
  std::string hltTag_;
  std::string processname_;

public:
  explicit HLTInclusiveVBFClient(const edm::ParameterSet&);
  ~HLTInclusiveVBFClient() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {}
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c) override;
  virtual void runClient_();
};

#endif
