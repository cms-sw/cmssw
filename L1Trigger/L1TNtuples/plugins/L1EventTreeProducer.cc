// -*- C++ -*-
//
// Package:    L1TriggerDPG/L1Ntuples
// Class:      L1EventTreeProducer
//
/**\class L1EventTreeProducer L1EventTreeProducer.cc L1TriggerDPG/L1Ntuples/src/L1EventTreeProducer.cc

Description: Produce L1 Extra tree

Implementation:
     
*/
//
// Original Author:
//         Created:
// $Id: L1EventTreeProducer.cc,v 1.8 2012/08/29 12:44:03 jbrooke Exp $
//
//

// system include files
#include <memory>

// framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// data formats
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

#include "L1Trigger/L1TNtuples/interface/L1AnalysisEvent.h"

//
// class declaration
//

class L1EventTreeProducer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit L1EventTreeProducer(const edm::ParameterSet&);
  ~L1EventTreeProducer() override = default;

private:
  void beginJob(void) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

public:
  std::unique_ptr<L1Analysis::L1AnalysisEvent> l1Event;
  L1Analysis::L1AnalysisEventDataFormat* l1EventData;

private:
  // output file
  edm::Service<TFileService> fs_;

  // tree
  TTree* tree_;

  // EDM input tags
  const edm::EDGetTokenT<edm::TriggerResults> hltSource_;
};

L1EventTreeProducer::L1EventTreeProducer(const edm::ParameterSet& iConfig)
    : hltSource_(consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("hltSource"))) {
  std::string puMCFile = iConfig.getUntrackedParameter<std::string>("puMCFile", "");
  std::string puMCHist = iConfig.getUntrackedParameter<std::string>("puMCHist", "pileup");
  std::string puDataFile = iConfig.getUntrackedParameter<std::string>("puDataFile", "");
  std::string puDataHist = iConfig.getUntrackedParameter<std::string>("puDataHist", "pileup");

  usesResource(TFileService::kSharedResource);

  bool useAvgVtx = iConfig.getUntrackedParameter<bool>("useAvgVtx", true);
  double maxAllowedWeight = iConfig.getUntrackedParameter<double>("maxAllowedWeight", -1);

  l1Event = std::make_unique<L1Analysis::L1AnalysisEvent>(
      puMCFile, puMCHist, puDataFile, puDataHist, useAvgVtx, maxAllowedWeight, consumesCollector());
  l1EventData = l1Event->getData();

  // set up output
  tree_ = fs_->make<TTree>("L1EventTree", "L1EventTree");
  tree_->Branch("Event", "L1Analysis::L1AnalysisEventDataFormat", &l1EventData, 32000, 3);
}

//
// member functions
//

// ------------ method called to for each event  ------------
void L1EventTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (!hltSource_.isUninitialized()) {
    l1Event->Reset();
    l1Event->Set(iEvent, hltSource_);
  }
  tree_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void L1EventTreeProducer::beginJob(void) {}

// ------------ method called once each job just after ending the event loop  ------------
void L1EventTreeProducer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(L1EventTreeProducer);
