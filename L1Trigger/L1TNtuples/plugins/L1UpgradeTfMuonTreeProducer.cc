// -*- C++ -*-
//
// Package:    L1Trigger/L1Ntuples
// Class:      L1UpgradeTfMuonTreeProducer
// 
/**\class L1UpgradeTfMuonTreeProducer L1UpgradeTfMuonTreeProducer.cc L1Trigger/L1Ntuples/plugins/L1UpgradeTfMuonTreeProducer.cc

Implementation:
     
*/
//
// Original Author:  
//         Created:  
// $Id: L1UpgradeTfMuonTreeProducer.cc,v 1.8 2012/08/29 12:44:03 jbrooke Exp $
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

// data formats
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1UpgradeTfMuon.h"
#include "L1Trigger/L1TNtuples/interface/L1AnalysisBMTFInputs.h"

//
// class declaration
//

class L1UpgradeTfMuonTreeProducer : public edm::EDAnalyzer {
public:
  explicit L1UpgradeTfMuonTreeProducer(const edm::ParameterSet&);
  ~L1UpgradeTfMuonTreeProducer();
  
  
private:
  virtual void beginJob(void) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

public:
  
  L1Analysis::L1AnalysisL1UpgradeTfMuon l1UpgradeBmtf;
  L1Analysis::L1AnalysisL1UpgradeTfMuon l1UpgradeOmtf;
  L1Analysis::L1AnalysisL1UpgradeTfMuon l1UpgradeEmtf;
  L1Analysis::L1AnalysisBMTFInputs  l1UpgradeBmtfInputs;
  L1Analysis::L1AnalysisL1UpgradeTfMuonDataFormat * l1UpgradeBmtfData; 
  L1Analysis::L1AnalysisL1UpgradeTfMuonDataFormat * l1UpgradeOmtfData;
  L1Analysis::L1AnalysisL1UpgradeTfMuonDataFormat * l1UpgradeEmtfData;
  L1Analysis::L1AnalysisBMTFInputsDataFormat * l1UpgradeBmtfInputsData;

private:

  unsigned maxL1UpgradeTfMuon_;

  // output file
  edm::Service<TFileService> fs_;
  
  // tree
  TTree * tree_;
 
  // EDM input tags
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> bmtfMuonToken_;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> omtfMuonToken_;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> emtfMuonToken_;

  edm::EDGetTokenT<L1MuDTChambPhContainer> bmtfPhInputToken_;
  edm::EDGetTokenT<L1MuDTChambThContainer> bmtfThInputToken_;

};



L1UpgradeTfMuonTreeProducer::L1UpgradeTfMuonTreeProducer(const edm::ParameterSet& iConfig)
{

  bmtfMuonToken_ = consumes<l1t::RegionalMuonCandBxCollection>(iConfig.getUntrackedParameter<edm::InputTag>("bmtfMuonToken"));
  omtfMuonToken_ = consumes<l1t::RegionalMuonCandBxCollection>(iConfig.getUntrackedParameter<edm::InputTag>("omtfMuonToken"));
  emtfMuonToken_ = consumes<l1t::RegionalMuonCandBxCollection>(iConfig.getUntrackedParameter<edm::InputTag>("emtfMuonToken"));

  bmtfPhInputToken_ = consumes<L1MuDTChambPhContainer>(iConfig.getUntrackedParameter<edm::InputTag>("bmtfInputPhMuonToken"));
  bmtfThInputToken_ = consumes<L1MuDTChambThContainer>(iConfig.getUntrackedParameter<edm::InputTag>("bmtfInputThMuonToken"));

  maxL1UpgradeTfMuon_ = iConfig.getParameter<unsigned int>("maxL1UpgradeTfMuon");
 

  l1UpgradeBmtfData = l1UpgradeBmtf.getData();
  l1UpgradeOmtfData = l1UpgradeOmtf.getData();
  l1UpgradeEmtfData = l1UpgradeEmtf.getData();
  l1UpgradeBmtfInputsData = l1UpgradeBmtfInputs.getData();

  
  // set up output
  tree_=fs_->make<TTree>("L1UpgradeTfMuonTree", "L1UpgradeTfMuonTree");
  tree_->Branch("L1UpgradeBmtfMuon", "L1Analysis::L1AnalysisL1UpgradeTfMuonDataFormat", &l1UpgradeBmtfData, 32000, 3);
  tree_->Branch("L1UpgradeOmtfMuon", "L1Analysis::L1AnalysisL1UpgradeTfMuonDataFormat", &l1UpgradeOmtfData, 32000, 3);
  tree_->Branch("L1UpgradeEmtfMuon", "L1Analysis::L1AnalysisL1UpgradeTfMuonDataFormat", &l1UpgradeEmtfData, 32000, 3);

  tree_->Branch("L1UpgradeBmtfInputs", "L1Analysis::L1AnalysisBMTFInputsDataFormat", &l1UpgradeBmtfInputsData, 32000, 3);


}


L1UpgradeTfMuonTreeProducer::~L1UpgradeTfMuonTreeProducer()
{
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1UpgradeTfMuonTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  l1UpgradeBmtf.Reset();
  l1UpgradeOmtf.Reset();
  l1UpgradeEmtf.Reset();
  l1UpgradeBmtfInputs.Reset();


  edm::Handle<l1t::RegionalMuonCandBxCollection> bmtfMuon;
  edm::Handle<l1t::RegionalMuonCandBxCollection> omtfMuon;
  edm::Handle<l1t::RegionalMuonCandBxCollection> emtfMuon;
  edm::Handle<L1MuDTChambPhContainer> bmtfPhInputs;
  edm::Handle<L1MuDTChambThContainer> bmtfThInputs;

  iEvent.getByToken(bmtfMuonToken_, bmtfMuon);
  iEvent.getByToken(omtfMuonToken_, omtfMuon);
  iEvent.getByToken(emtfMuonToken_, emtfMuon);
  iEvent.getByToken(bmtfPhInputToken_, bmtfPhInputs);
  iEvent.getByToken(bmtfThInputToken_, bmtfThInputs);

  if (bmtfMuon.isValid()){ 
    l1UpgradeBmtf.SetTfMuon(*bmtfMuon, maxL1UpgradeTfMuon_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade BMTF muons not found. Branch will not be filled" << std::endl;
  }

  if (omtfMuon.isValid()){ 
    l1UpgradeOmtf.SetTfMuon(*omtfMuon, maxL1UpgradeTfMuon_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade OMTF muons not found. Branch will not be filled" << std::endl;
  }

  if (emtfMuon.isValid()){ 
    l1UpgradeEmtf.SetTfMuon(*emtfMuon, maxL1UpgradeTfMuon_);
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade EMTF muons not found. Branch will not be filled" << std::endl;
  }


  int max_inputs = maxL1UpgradeTfMuon_*4;

  if (!bmtfPhInputs.isValid()) {
    edm::LogWarning("MissingProduct") << "L1Upgrade BMTF Ph Inputs not found. Branch will not be filled" << std::endl;
    }
  else l1UpgradeBmtfInputs.SetBMPH(bmtfPhInputs, max_inputs);

  if (!bmtfThInputs.isValid()) {
    edm::LogWarning("MissingProduct") << "L1Upgrade BMTF Th Inputs not found. Branch will not be filled" << std::endl;
    }
  else l1UpgradeBmtfInputs.SetBMTH(bmtfThInputs, max_inputs);



  tree_->Fill();

}

// ------------ method called once each job just before starting event loop  ------------
void 
L1UpgradeTfMuonTreeProducer::beginJob(void)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1UpgradeTfMuonTreeProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1UpgradeTfMuonTreeProducer);
