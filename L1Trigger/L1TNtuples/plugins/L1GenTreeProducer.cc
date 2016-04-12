// -*- C++ -*-
//
// Package:    L1TriggerDPG/L1Ntuples
// Class:      L1GenTreeProducer
// 
/**\class L1GenTreeProducer L1GenTreeProducer.cc L1TriggerDPG/L1Ntuples/src/L1GenTreeProducer.cc

Description: Produce L1 Extra tree

Implementation:
     
*/
//
// Original Author:  
//         Created:  
// $Id: L1GenTreeProducer.cc,v 1.8 2012/08/29 12:44:03 jbrooke Exp $
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

// input data formats
#include "DataFormats/JetReco/interface/GenJetCollection.h"


// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1GeneratorDataFormat.h"

//
// class declaration
//

class L1GenTreeProducer : public edm::EDAnalyzer {
public:
  explicit L1GenTreeProducer(const edm::ParameterSet&);
  ~L1GenTreeProducer();
  
  
private:
  virtual void beginJob(void) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();  

private:

  unsigned maxL1Upgrade_;

  // output file
  edm::Service<TFileService> fs_;
  
  // tree
  TTree * tree_;
 
  // data format
  L1Analysis::L1AnalysisL1GeneratorDataFormat * l1GenData_;

  // EDM input tags
  edm::EDGetTokenT<reco::GenJetCollection> genJetToken_;


};



L1GenTreeProducer::L1GenTreeProducer(const edm::ParameterSet& iConfig)
{

  genJetToken_ = consumes<reco::GenJetCollection>(iConfig.getUntrackedParameter<edm::InputTag>("genJetToken"));
  
  // set up output
  tree_=fs_->make<TTree>("L1GenTree", "L1GenTree");
  tree_->Branch("Generator", "L1Analysis::L1AnalysisL1GeneratorDataFormat", &l1GenData, 32000, 3);

}


L1GenTreeProducer::~L1GenTreeProducer()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1GenTreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  l1GenData_->Reset();

  edm::Handle<reco::GenJetCollection> genJets;

  iEvent.getByToken(genJetToken_, genJets);


  if (genJets.isValid()){ 

    reco::GenJetCollection::const_iterator jetItr = genJets->begin();
    reco::GenJetCollection::const_iterator jetEnd = genJets->end();
    for( ; jetItr != jetEnd ; ++jetItr) {
      l1GenData_->jetPt->push_back( jetItr->pt() );
      l1GenData_->jetEta->push_back( jetItr->eta() );
      l1GenData_->jetPhi->push_back( jetItr->phi() );
      l1GenData_->nJet++;
    }

  } else {
    edm::LogWarning("MissingProduct") << "Gen jets not found. Branch will not be filled" << std::endl;
  }

  tree_->Fill();

}

// ------------ method called once each job just before starting event loop  ------------
void 
L1GenTreeProducer::beginJob(void)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1GenTreeProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1GenTreeProducer);
