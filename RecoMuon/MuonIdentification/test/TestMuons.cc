// -*- C++ -*-
//
// Original Author:  Dmytro Kovalskyi
// $Id: TestMuons.cc,v 1.3 2007/07/17 00:12:42 dmytro Exp $
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/MuonReco/interface/Muon.h"

class TestMuons : public edm::EDAnalyzer {
 public:
   explicit TestMuons(const edm::ParameterSet&);
   virtual ~TestMuons(){}
   
   virtual void analyze (const edm::Event&, const edm::EventSetup&);

 private:
   edm::InputTag theInputCollection;
};

TestMuons::TestMuons(const edm::ParameterSet& iConfig)
{
   theInputCollection = iConfig.getParameter<edm::InputTag>("InputCollection");
}

void TestMuons::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // Handle<reco::MuonCollection> muons;
   edm::Handle<edm::View<reco::Muon> > muons;
   iEvent.getByLabel(theInputCollection, muons);
   
   for(edm::View<reco::Muon>::const_iterator muon = muons->begin(); muon != muons->end(); ++muon){
      std::cout << "\n----------------------------------------------------" << std::endl;
      std::cout << "Muon (pt,eta,phi): " << muon->pt() << ", " << muon->eta() << ", " << muon->phi() << std::endl;
      std::cout << "\t energy (ecal, hcal, ho): " << muon->calEnergy().em << ", " << 
	muon->calEnergy().had << ", " << muon->calEnergy().ho << std::endl;
      std::cout << "\t isolation dR=0.3 (sumPt, emEt, hadEt, hoEt, nTracks, nJets): " << 
	muon->isolationR03().sumPt << ", " << muon->isolationR03().emEt << ", " << 
	muon->isolationR03().hadEt << ", " << muon->isolationR03().hoEt << ", " <<
	muon->isolationR03().nTracks << ", " << muon->isolationR03().nJets << std::endl;
      std::cout << "\t isolation dR=0.5 (sumPt, emEt, hadEt, hoEt, nTracks, nJets): " << 
	muon->isolationR05().sumPt << ", " << muon->isolationR05().emEt << ", " << 
	muon->isolationR05().hadEt << ", " << muon->isolationR05().hoEt << ", " <<
	muon->isolationR05().nTracks << ", " << muon->isolationR05().nJets << std::endl;
      std::cout << "\t # matches: " << muon->numberOfMatches() << std::endl;
      std::cout << "\t # caloCompatibility: " << muon->caloCompatibility() << std::endl;
      
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestMuons);
