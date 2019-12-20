// -*- C++ -*-
//
// Package:    L1Trigger/L1TNtuples
// Class:      L1ExtraTreeProducer
//
/**\class L1ExtraTreeProducer L1ExtraTreeProducer.cc L1Trigger/L1TNtuples/src/L1ExtraTreeProducer.cc

Description: Produce L1 Extra tree

Implementation:
     
*/
//
// Original Author:  Alex Tapper
//         Created:
// $Id: L1ExtraTreeProducer.cc,v 1.6 2010/06/17 20:44:45 econte Exp $
//
//

// system include files
#include <memory>

// framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// L1Gt Utils
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"

// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1Menu.h"

//
// class declaration
//

class L1MenuTreeProducer : public edm::EDAnalyzer {
public:
  explicit L1MenuTreeProducer(const edm::ParameterSet&);
  ~L1MenuTreeProducer() override;

private:
  void beginJob(void) override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override {}
  void endJob() override;

public:
  L1Analysis::L1AnalysisL1Menu* l1Menu;
  L1Analysis::L1AnalysisL1MenuDataFormat* l1MenuData;

private:
  // output file
  edm::Service<TFileService> fs_;

  // tree
  TTree* tree_;

  // EDM input tags
  //  edm::InputTag l1MenuInputTag_;
  L1GtUtils l1GtUtils_;
};

L1MenuTreeProducer::L1MenuTreeProducer(const edm::ParameterSet& iConfig)
    :  //  l1MenuInputTag_(iConfig.getParameter<edm::InputTag>("L1MenuInputTag")),
      l1GtUtils_(iConfig, consumesCollector(), true, L1GtUtils::UseEventSetupIn::Event) {
  l1Menu = new L1Analysis::L1AnalysisL1Menu();
  l1MenuData = l1Menu->getData();

  // set up output
  tree_ = fs_->make<TTree>("L1MenuTree", "L1MenuTree");
  tree_->Branch("L1Menu", "L1Analysis::L1AnalysisL1MenuDataFormat", &l1MenuData, 32000, 0 /*3*/);
}

L1MenuTreeProducer::~L1MenuTreeProducer() {}

//
// member functions
//

// ------------ method called to for each event  ------------
void L1MenuTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  l1Menu->Reset();

  // getting l1GtUtils
  edm::LogInfo("L1TNtuples") << "Extracting menu ... " << std::endl;
  l1GtUtils_.retrieveL1EventSetup(iSetup);

  // testing menu
  edm::LogInfo("L1TNtuples") << "L1 trigger menu name and implementation:"
                             << "\n"
                             << l1GtUtils_.l1TriggerMenu() << "\n"
                             << l1GtUtils_.l1TriggerMenuImplementation() << std::endl;

  // adding PrescaleFactorIndex
  l1Menu->SetPrescaleFactorIndex(l1GtUtils_, iEvent);

  tree_->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void L1MenuTreeProducer::beginJob(void) {}

void L1MenuTreeProducer::beginRun(const edm::Run& iRun, const edm::EventSetup& evSetup) {
  // L1GtTriggerMenuLite input tag from provenance
  //l1GtUtils_.getL1GtRunCache(iRun, evSetup, true, true);
}

// ------------ method called once each job just after ending the event loop  ------------
void L1MenuTreeProducer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(L1MenuTreeProducer);
