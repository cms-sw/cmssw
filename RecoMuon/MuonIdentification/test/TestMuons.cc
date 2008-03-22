// -*- C++ -*-
//
// Original Author:  Dmytro Kovalskyi
// $Id: TestMuons.cc,v 1.2 2007/07/16 23:57:33 dmytro Exp $
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
      std::cout << "\t energy (ecal, hcal, ho): " << muon->getCalEnergy().em << ", " << 
	muon->getCalEnergy().had << ", " << muon->getCalEnergy().ho << std::endl;
      std::cout << "\t isolation dR=0.3 (sumPt, emEt, hadEt, hoEt, nTracks, nJets): " << 
	muon->getIsolationR03().sumPt << ", " << muon->getIsolationR03().emEt << ", " << 
	muon->getIsolationR03().hadEt << ", " << muon->getIsolationR03().hoEt << ", " <<
	muon->getIsolationR03().nTracks << ", " << muon->getIsolationR03().nJets << std::endl;
      std::cout << "\t isolation dR=0.5 (sumPt, emEt, hadEt, hoEt, nTracks, nJets): " << 
	muon->getIsolationR05().sumPt << ", " << muon->getIsolationR05().emEt << ", " << 
	muon->getIsolationR05().hadEt << ", " << muon->getIsolationR05().hoEt << ", " <<
	muon->getIsolationR05().nTracks << ", " << muon->getIsolationR05().nJets << std::endl;
      std::cout << "\t # matches: " << muon->numberOfMatches() << std::endl;
      std::cout << "\t # caloCompatibility: " << muon->getCaloCompatibility() << std::endl;
      
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestMuons);
