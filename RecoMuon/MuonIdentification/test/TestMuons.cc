#ifndef RecoMuon_MuonIdentification_TestMuon_H
#define RecoMuon_MuonIdentification_TestMuon_H

/** \class TestMuon
 *  Producer meant for the test of the muon object and the maps associated to it.
 *
 * 
 *
 *  $Date: 2011/11/02 06:50:24 $
 *  $Revision: 1.7 $
 *  \author Dmytro Kovalskyi, R. Bellan - UCSB <riccardo.bellan@cern.ch> 
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"

class TestMuons : public edm::EDAnalyzer {
 public:
   explicit TestMuons(const edm::ParameterSet&);
   virtual ~TestMuons(){}
   
  virtual void analyze (const edm::Event&, const edm::EventSetup&);
  void printMuonCollections(const edm::Handle<edm::View<reco::Muon> > &muons);
  void checkTimeMaps(const edm::Event& iEvent, const edm::Handle<reco::MuonCollection> &muons);
  void checkPFMap(const edm::Event& iEvent, const edm::Handle<reco::MuonCollection> &muons);
 private:
  edm::InputTag theInput;

};
#endif

TestMuons::TestMuons(const edm::ParameterSet& iConfig){

  theInput = iConfig.getParameter<edm::InputTag>("InputCollection");

}

void TestMuons::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup){

  edm::Handle<edm::View<reco::Muon> > muonsV;
  iEvent.getByLabel(theInput, muonsV);   
  printMuonCollections(muonsV);
 
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByLabel(theInput, muons);
  
  checkTimeMaps(iEvent,muons);
  checkPFMap(iEvent,muons);
}

void TestMuons::checkTimeMaps(const edm::Event& iEvent, const edm::Handle<reco::MuonCollection> &muons){

  std::cout << "checkTimeMaps"  <<std::endl;

  edm::Handle<reco::MuonTimeExtraMap> timeMapCmb;
  edm::Handle<reco::MuonTimeExtraMap> timeMapDT;
  edm::Handle<reco::MuonTimeExtraMap> timeMapCSC;

  iEvent.getByLabel(theInput.label(),"combined",timeMapCmb);
  iEvent.getByLabel(theInput.label(),"dt",timeMapDT);
  iEvent.getByLabel(theInput.label(),"csc",timeMapCSC);
  
  for(unsigned int imucount=0; imucount < muons->size(); ++imucount){
    reco::MuonRef muonR(muons,imucount);
    std::cout  << "Ref: " << muonR.id() << " " << muonR.key()
	       << " Time CMB " <<  (*timeMapCmb)[muonR].timeAtIpInOut()
	       << " time DT "  <<  (*timeMapDT)[muonR].timeAtIpInOut()
	       << " time CSC " <<  (*timeMapCSC)[muonR].timeAtIpInOut() << std::endl;
  }
}





void TestMuons::printMuonCollections(const edm::Handle<edm::View<reco::Muon> > &muons){
  

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

    
    if(muon->isAValidMuonTrack(reco::Muon::TPFMS))
      std::cout << "TPFMS pt: " << muon->tpfmsTrack()->pt() << std::endl;     
    
    if(muon->isAValidMuonTrack(reco::Muon::Picky))
      std::cout << "Picky pt: " << muon->pickyTrack()->pt() << std::endl;     
    
    if(muon->isAValidMuonTrack(reco::Muon::DYT))
      std::cout << "DYT pt: " << muon->dytTrack()->pt() << std::endl;     
   
    if(muon->isPFIsolationValid()){
      std::cout << "PF Isolation is Valid." << std::endl 
		<< "Iso 0.3, (sumChargedHadronPt, sumChargedParticlePt, sumNeutralHadronEt, sumPhotonEt, sumNeutralHadronEtHighThreshold, sumPhotonEtHighThreshold, sumPUPt): "
		<< muon->pfIsolationR03().sumChargedHadronPt << ", " << muon->pfIsolationR03().sumChargedParticlePt << ", " 
		<< muon->pfIsolationR03().sumNeutralHadronEt << ", " << muon->pfIsolationR03().sumPhotonEt << ", " 
		<< muon->pfIsolationR03().sumNeutralHadronEtHighThreshold << ", " << muon->pfIsolationR03().sumPhotonEtHighThreshold << ", " 
		<< muon->pfIsolationR03().sumPUPt <<std::endl;
      std::cout << "Iso 0.4, (sumChargedHadronPt, sumChargedParticlePt, sumNeutralHadronEt, sumPhotonEt, sumNeutralHadronEtHighThreshold, sumPhotonEtHighThreshold, sumPUPt): "
		<< muon->pfIsolationR04().sumChargedHadronPt << ", " << muon->pfIsolationR04().sumChargedParticlePt << ", " 
		<< muon->pfIsolationR04().sumNeutralHadronEt << ", " << muon->pfIsolationR04().sumPhotonEt << ", " 
		<< muon->pfIsolationR04().sumNeutralHadronEtHighThreshold << ", " << muon->pfIsolationR04().sumPhotonEtHighThreshold << ", " 
		<< muon->pfIsolationR04().sumPUPt <<std::endl;


    }

 
  }
  
}


void TestMuons::checkPFMap(const edm::Event& iEvent, const edm::Handle<reco::MuonCollection> &muons){

  std::cout << "checkPFMaps"  <<std::endl;

  edm::Handle<edm::ValueMap<reco::PFCandidatePtr> > pfMap;
  iEvent.getByLabel("particleFlow",theInput.label(),pfMap);
  
  
  for(unsigned int imucount=0; imucount < muons->size(); ++imucount){
    reco::MuonRef muonR(muons,imucount);
    if ( (*pfMap)[muonR].isNonnull())
      std::cout  << "PfCand muon pT: " << (*pfMap)[muonR]->muonRef()->pt()
		 << " Muon pT: " << muonR->pt() << std::endl; 
  }
}




//define this as a plug-in
DEFINE_FWK_MODULE(TestMuons);
