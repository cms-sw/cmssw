#ifndef PlotMaker_h
#define PlotMaker_h

/*  \class PlotMaker
*
*  Class to produce some plots of Off-line variables in the TriggerValidation Code
*
*  Author: Massimiliano Chiorboli      Date: September 2007
*
*/
#include <memory>
#include <string>
#include <iostream>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"


#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"

#include "TH1.h"
#include "TH2.h"

class PlotMaker {

 public:
  PlotMaker(edm::ParameterSet objectList);
  virtual ~PlotMaker(){};

  void handleObjects(const edm::Event&);
  void fillPlots(const edm::Event&);
  void bookHistos(std::vector<int>*, std::vector<int>*,std::vector<std::string>*,std::vector<std::string>*);



 private:

  void setBits(std::vector<int>* l1bits, std::vector<int>* hltbits) {l1bits_=l1bits; hltbits_=hltbits;}
  std::vector<int>* l1bits_;
  std::vector<int>* hltbits_;
    
    

  // Define the parameters
  std::string m_l1extra;
  std::string m_electronSrc;
  std::string m_muonSrc;
  std::string m_jetsSrc;
  std::string m_photonSrc;
  std::string m_calometSrc;


  double def_electronPtMin; 
  double def_muonPtMin    ; 
  double def_jetPtMin     ; 
  double def_photonPtMin  ; 
  
  reco::PixelMatchGsfElectronCollection theElectronCollection;
  reco::MuonCollection                  theMuonCollection    ;
  reco::PhotonCollection                thePhotonCollection  ;
  reco::CaloJetCollection               theCaloJetCollection ;
  reco::CaloMETCollection               theCaloMETCollection ;

  l1extra::L1EmParticleCollection theL1EmIsoCollection, theL1EmNotIsoCollection;
  l1extra::L1MuonParticleCollection theL1MuonCollection;
  l1extra::L1JetParticleCollection theL1CentralJetCollection, theL1ForwardJetCollection, theL1TauJetCollection;
  l1extra::L1EtMissParticle theL1METCollection;

  //histos

  //Jets
  TH1D* hL1CentralJetMult;
  TH1D* hL1ForwardJetMult;
  TH1D* hL1TauJetMult;
  TH1D* hJetMult;
  std::vector<TH1D*> hL1CentralJetMultAfterL1;
  std::vector<TH1D*> hL1CentralJetMultAfterHLT;
  std::vector<TH1D*> hL1ForwardJetMultAfterL1;
  std::vector<TH1D*> hL1ForwardJetMultAfterHLT;
  std::vector<TH1D*> hL1TauJetMultAfterL1;
  std::vector<TH1D*> hL1TauJetMultAfterHLT;
  std::vector<TH1D*> hJetMultAfterL1;
  std::vector<TH1D*> hJetMultAfterHLT;
  TH1D* hL1CentralJet1Pt;
  TH1D* hL1ForwardJet1Pt;
  TH1D* hL1TauJet1Pt;
  TH1D* hJet1Pt;
  std::vector<TH1D*> hL1CentralJet1PtAfterL1;
  std::vector<TH1D*> hL1CentralJet1PtAfterHLT;
  std::vector<TH1D*> hL1ForwardJet1PtAfterL1;
  std::vector<TH1D*> hL1ForwardJet1PtAfterHLT;
  std::vector<TH1D*> hL1TauJet1PtAfterL1;
  std::vector<TH1D*> hL1TauJet1PtAfterHLT;
  std::vector<TH1D*> hJet1PtAfterL1;
  std::vector<TH1D*> hJet1PtAfterHLT;
  TH1D* hL1CentralJet2Pt;
  TH1D* hL1ForwardJet2Pt;
  TH1D* hL1TauJet2Pt;
  TH1D* hJet2Pt;
  std::vector<TH1D*> hL1CentralJet2PtAfterL1;
  std::vector<TH1D*> hL1CentralJet2PtAfterHLT;
  std::vector<TH1D*> hL1ForwardJet2PtAfterL1;
  std::vector<TH1D*> hL1ForwardJet2PtAfterHLT;
  std::vector<TH1D*> hL1TauJet2PtAfterL1;
  std::vector<TH1D*> hL1TauJet2PtAfterHLT;
  std::vector<TH1D*> hJet2PtAfterL1;
  std::vector<TH1D*> hJet2PtAfterHLT;
  TH1D* hL1CentralJet1Eta;
  TH1D* hL1ForwardJet1Eta;
  TH1D* hL1TauJet1Eta;
  TH1D* hJet1Eta;
  std::vector<TH1D*> hL1CentralJet1EtaAfterL1;
  std::vector<TH1D*> hL1CentralJet1EtaAfterHLT;
  std::vector<TH1D*> hL1ForwardJet1EtaAfterL1;
  std::vector<TH1D*> hL1ForwardJet1EtaAfterHLT;
  std::vector<TH1D*> hL1TauJet1EtaAfterL1;
  std::vector<TH1D*> hL1TauJet1EtaAfterHLT;
  std::vector<TH1D*> hJet1EtaAfterL1;
  std::vector<TH1D*> hJet1EtaAfterHLT;
  TH1D* hL1CentralJet2Eta;
  TH1D* hL1ForwardJet2Eta;
  TH1D* hL1TauJet2Eta;
  TH1D* hJet2Eta;
  std::vector<TH1D*> hL1CentralJet2EtaAfterL1;
  std::vector<TH1D*> hL1CentralJet2EtaAfterHLT;
  std::vector<TH1D*> hL1ForwardJet2EtaAfterL1;
  std::vector<TH1D*> hL1ForwardJet2EtaAfterHLT;
  std::vector<TH1D*> hL1TauJet2EtaAfterL1;
  std::vector<TH1D*> hL1TauJet2EtaAfterHLT;
  std::vector<TH1D*> hJet2EtaAfterL1;
  std::vector<TH1D*> hJet2EtaAfterHLT;

  //Electrons
  TH1D* hL1EmIsoMult;
  TH1D* hL1EmNotIsoMult;
  TH1D* hElecMult;
  std::vector<TH1D*> hL1EmIsoMultAfterL1;
  std::vector<TH1D*> hL1EmIsoMultAfterHLT;
  std::vector<TH1D*> hL1EmNotIsoMultAfterL1;
  std::vector<TH1D*> hL1EmNotIsoMultAfterHLT;
  std::vector<TH1D*> hElecMultAfterL1;
  std::vector<TH1D*> hElecMultAfterHLT;
  TH1D* hL1EmIso1Pt;
  TH1D* hL1EmNotIso1Pt;
  TH1D* hElec1Pt;
  std::vector<TH1D*> hL1EmIso1PtAfterL1;
  std::vector<TH1D*> hL1EmIso1PtAfterHLT;
  std::vector<TH1D*> hL1EmNotIso1PtAfterL1;
  std::vector<TH1D*> hL1EmNotIso1PtAfterHLT;
  std::vector<TH1D*> hElec1PtAfterL1;
  std::vector<TH1D*> hElec1PtAfterHLT;
  TH1D* hL1EmIso2Pt;
  TH1D* hL1EmNotIso2Pt;
  TH1D* hElec2Pt;
  std::vector<TH1D*> hL1EmIso2PtAfterL1;
  std::vector<TH1D*> hL1EmIso2PtAfterHLT;
  std::vector<TH1D*> hL1EmNotIso2PtAfterL1;
  std::vector<TH1D*> hL1EmNotIso2PtAfterHLT;
  std::vector<TH1D*> hElec2PtAfterL1;
  std::vector<TH1D*> hElec2PtAfterHLT;
  TH1D* hL1EmIso1Eta;
  TH1D* hL1EmNotIso1Eta;
  TH1D* hElec1Eta;
  std::vector<TH1D*> hL1EmIso1EtaAfterL1;
  std::vector<TH1D*> hL1EmIso1EtaAfterHLT;
  std::vector<TH1D*> hL1EmNotIso1EtaAfterL1;
  std::vector<TH1D*> hL1EmNotIso1EtaAfterHLT;
  std::vector<TH1D*> hElec1EtaAfterL1;
  std::vector<TH1D*> hElec1EtaAfterHLT;
  TH1D* hL1EmIso2Eta;
  TH1D* hL1EmNotIso2Eta;
  TH1D* hElec2Eta;
  std::vector<TH1D*> hL1EmIso2EtaAfterL1;
  std::vector<TH1D*> hL1EmIso2EtaAfterHLT;
  std::vector<TH1D*> hL1EmNotIso2EtaAfterL1;
  std::vector<TH1D*> hL1EmNotIso2EtaAfterHLT;
  std::vector<TH1D*> hElec2EtaAfterL1;
  std::vector<TH1D*> hElec2EtaAfterHLT;
  
  //Muons
  TH1D* hL1MuonMult;
  TH1D* hMuonMult;
  std::vector<TH1D*> hL1MuonMultAfterL1;
  std::vector<TH1D*> hL1MuonMultAfterHLT;
  std::vector<TH1D*> hMuonMultAfterL1;
  std::vector<TH1D*> hMuonMultAfterHLT;
  TH1D* hL1Muon1Pt;
  TH1D* hMuon1Pt;
  std::vector<TH1D*> hL1Muon1PtAfterL1;
  std::vector<TH1D*> hL1Muon1PtAfterHLT;
  std::vector<TH1D*> hMuon1PtAfterL1;
  std::vector<TH1D*> hMuon1PtAfterHLT;
  TH1D* hL1Muon2Pt;
  TH1D* hMuon2Pt;
  std::vector<TH1D*> hL1Muon2PtAfterL1;
  std::vector<TH1D*> hL1Muon2PtAfterHLT;
  std::vector<TH1D*> hMuon2PtAfterL1;
  std::vector<TH1D*> hMuon2PtAfterHLT;
  TH1D* hL1Muon1Eta;
  TH1D* hMuon1Eta;
  std::vector<TH1D*> hL1Muon1EtaAfterL1;
  std::vector<TH1D*> hL1Muon1EtaAfterHLT;
  std::vector<TH1D*> hMuon1EtaAfterL1;
  std::vector<TH1D*> hMuon1EtaAfterHLT;
  TH1D* hL1Muon2Eta;
  TH1D* hMuon2Eta;
  std::vector<TH1D*> hL1Muon2EtaAfterL1;
  std::vector<TH1D*> hL1Muon2EtaAfterHLT;
  std::vector<TH1D*> hMuon2EtaAfterL1;
  std::vector<TH1D*> hMuon2EtaAfterHLT;


  //Photons
  TH1D* hPhotonMult;
  std::vector<TH1D*> hPhotonMultAfterL1;
  std::vector<TH1D*> hPhotonMultAfterHLT;
  TH1D* hPhoton1Pt;
  std::vector<TH1D*> hPhoton1PtAfterL1;
  std::vector<TH1D*> hPhoton1PtAfterHLT;
  TH1D* hPhoton2Pt;
  std::vector<TH1D*> hPhoton2PtAfterL1;
  std::vector<TH1D*> hPhoton2PtAfterHLT;
  TH1D* hPhoton1Eta;
  std::vector<TH1D*> hPhoton1EtaAfterL1;
  std::vector<TH1D*> hPhoton1EtaAfterHLT;
  TH1D* hPhoton2Eta;
  std::vector<TH1D*> hPhoton2EtaAfterL1;
  std::vector<TH1D*> hPhoton2EtaAfterHLT;
  

  
  //MET
  TH1D* hL1MET;
  TH1D* hMET;
  std::vector<TH1D*> hL1METAfterL1;
  std::vector<TH1D*> hL1METAfterHLT;
  std::vector<TH1D*> hMETAfterL1;
  std::vector<TH1D*> hMETAfterHLT;






  string myHistoName;
  string myHistoTitle;

};


#endif
