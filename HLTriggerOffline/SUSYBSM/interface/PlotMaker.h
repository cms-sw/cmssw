#ifndef PlotMaker_h
#define PlotMaker_h

/*  \class PlotMaker
*
*  Class to produce some plots of Off-line variables in the TriggerValidation Code
*
*  Author: Massimiliano Chiorboli      Date: September 2007
//         Maurizio Pierini
//         Maria Spiropulu
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
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"


//l1extra
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"

//#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"


#include "TH1.h"
#include "TH2.h"

class PlotMaker {

 public:
  PlotMaker(edm::ParameterSet objectList);
  virtual ~PlotMaker(){};

  void handleObjects(const edm::Event&);
  void fillPlots(const edm::Event&);
  void bookHistos(std::vector<int>*, std::vector<int>*,std::vector<std::string>*,std::vector<std::string>*);
  void writeHistos();


 private:

  

  void setBits(std::vector<int>* l1bits, std::vector<int>* hltbits) {l1bits_=l1bits; hltbits_=hltbits;}
  double invariantMass(reco::Particle*,reco::Particle*);
  std::vector<int>* l1bits_;
  std::vector<int>* hltbits_;
    
    

  // Define the parameters
  std::string m_l1extra;
  std::string m_electronSrc;
  std::string m_muonSrc;
  std::string m_jetsSrc;
  std::string m_photonSrc;
  std::string m_photonProducerSrc;
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
  l1extra::L1EtMissParticleCollection theL1METCollection;

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
  TH1D* hL1CentralJet1Phi;
  TH1D* hL1ForwardJet1Phi;
  TH1D* hL1TauJet1Phi;
  TH1D* hJet1Phi;
  std::vector<TH1D*> hL1CentralJet1PhiAfterL1;
  std::vector<TH1D*> hL1CentralJet1PhiAfterHLT;
  std::vector<TH1D*> hL1ForwardJet1PhiAfterL1;
  std::vector<TH1D*> hL1ForwardJet1PhiAfterHLT;
  std::vector<TH1D*> hL1TauJet1PhiAfterL1;
  std::vector<TH1D*> hL1TauJet1PhiAfterHLT;
  std::vector<TH1D*> hJet1PhiAfterL1;
  std::vector<TH1D*> hJet1PhiAfterHLT;
  TH1D* hL1CentralJet2Phi;
  TH1D* hL1ForwardJet2Phi;
  TH1D* hL1TauJet2Phi;
  TH1D* hJet2Phi;
  std::vector<TH1D*> hL1CentralJet2PhiAfterL1;
  std::vector<TH1D*> hL1CentralJet2PhiAfterHLT;
  std::vector<TH1D*> hL1ForwardJet2PhiAfterL1;
  std::vector<TH1D*> hL1ForwardJet2PhiAfterHLT;
  std::vector<TH1D*> hL1TauJet2PhiAfterL1;
  std::vector<TH1D*> hL1TauJet2PhiAfterHLT;
  std::vector<TH1D*> hJet2PhiAfterL1;
  std::vector<TH1D*> hJet2PhiAfterHLT;

  TH1D* hDiJetInvMass;
  std::vector<TH1D*> hDiJetInvMassAfterL1;
  std::vector<TH1D*> hDiJetInvMassAfterHLT;




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
  TH1D* hL1EmIso1Phi;
  TH1D* hL1EmNotIso1Phi;
  TH1D* hElec1Phi;
  std::vector<TH1D*> hL1EmIso1PhiAfterL1;
  std::vector<TH1D*> hL1EmIso1PhiAfterHLT;
  std::vector<TH1D*> hL1EmNotIso1PhiAfterL1;
  std::vector<TH1D*> hL1EmNotIso1PhiAfterHLT;
  std::vector<TH1D*> hElec1PhiAfterL1;
  std::vector<TH1D*> hElec1PhiAfterHLT;
  TH1D* hL1EmIso2Phi;
  TH1D* hL1EmNotIso2Phi;
  TH1D* hElec2Phi;
  std::vector<TH1D*> hL1EmIso2PhiAfterL1;
  std::vector<TH1D*> hL1EmIso2PhiAfterHLT;
  std::vector<TH1D*> hL1EmNotIso2PhiAfterL1;
  std::vector<TH1D*> hL1EmNotIso2PhiAfterHLT;
  std::vector<TH1D*> hElec2PhiAfterL1;
  std::vector<TH1D*> hElec2PhiAfterHLT;
  
  TH1D* hDiElecInvMass;
  std::vector<TH1D*> hDiElecInvMassAfterL1;
  std::vector<TH1D*> hDiElecInvMassAfterHLT;


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
  TH1D* hL1Muon1Phi;
  TH1D* hMuon1Phi;
  std::vector<TH1D*> hL1Muon1PhiAfterL1;
  std::vector<TH1D*> hL1Muon1PhiAfterHLT;
  std::vector<TH1D*> hMuon1PhiAfterL1;
  std::vector<TH1D*> hMuon1PhiAfterHLT;
  TH1D* hL1Muon2Phi;
  TH1D* hMuon2Phi;
  std::vector<TH1D*> hL1Muon2PhiAfterL1;
  std::vector<TH1D*> hL1Muon2PhiAfterHLT;
  std::vector<TH1D*> hMuon2PhiAfterL1;
  std::vector<TH1D*> hMuon2PhiAfterHLT;

  TH1D* hDiMuonInvMass;
  std::vector<TH1D*> hDiMuonInvMassAfterL1;
  std::vector<TH1D*> hDiMuonInvMassAfterHLT;


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
  TH1D* hPhoton1Phi;
  std::vector<TH1D*> hPhoton1PhiAfterL1;
  std::vector<TH1D*> hPhoton1PhiAfterHLT;
  TH1D* hPhoton2Phi;
  std::vector<TH1D*> hPhoton2PhiAfterL1;
  std::vector<TH1D*> hPhoton2PhiAfterHLT;
  
  TH1D* hDiPhotonInvMass;
  std::vector<TH1D*> hDiPhotonInvMassAfterL1;
  std::vector<TH1D*> hDiPhotonInvMassAfterHLT;

  
  //MET
  TH1D* hL1MET;
  TH1D* hMET;
  TH1D* hL1METphi;
  TH1D* hMETphi;
  TH1D* hL1METx;
  TH1D* hMETx;
  TH1D* hL1METy;
  TH1D* hMETy;
  TH1D* hL1SumEt;
  TH1D* hSumEt;
  TH1D* hL1METSignificance;
  TH1D* hMETSignificance;

  std::vector<TH1D*> hL1METAfterL1;
  std::vector<TH1D*> hL1METAfterHLT;
  std::vector<TH1D*> hMETAfterL1;
  std::vector<TH1D*> hMETAfterHLT;
  std::vector<TH1D*> hL1METphiAfterL1;
  std::vector<TH1D*> hL1METphiAfterHLT;
  std::vector<TH1D*> hMETphiAfterL1;
  std::vector<TH1D*> hMETphiAfterHLT;
  std::vector<TH1D*> hL1METxAfterL1;
  std::vector<TH1D*> hL1METxAfterHLT;
  std::vector<TH1D*> hMETxAfterL1;
  std::vector<TH1D*> hMETxAfterHLT;
  std::vector<TH1D*> hL1METyAfterL1;
  std::vector<TH1D*> hL1METyAfterHLT;
  std::vector<TH1D*> hMETyAfterL1;
  std::vector<TH1D*> hMETyAfterHLT;
  std::vector<TH1D*> hL1SumEtAfterL1;
  std::vector<TH1D*> hL1SumEtAfterHLT;
  std::vector<TH1D*> hSumEtAfterL1;
  std::vector<TH1D*> hSumEtAfterHLT;
  std::vector<TH1D*> hL1METSignificanceAfterL1;
  std::vector<TH1D*> hL1METSignificanceAfterHLT;
  std::vector<TH1D*> hMETSignificanceAfterL1;
  std::vector<TH1D*> hMETSignificanceAfterHLT;





  string myHistoName;
  string myHistoTitle;

};


#endif
