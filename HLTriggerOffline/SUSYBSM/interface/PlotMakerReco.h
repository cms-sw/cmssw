#ifndef PlotMakerReco_h
#define PlotMakerReco_h

/*  \class PlotMakerReco
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


#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
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

//included for DQM
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



class PlotMakerReco {

 public:
  PlotMakerReco(edm::ParameterSet objectList);
  virtual ~PlotMakerReco(){};

  void handleObjects(const edm::Event&);
  void fillPlots(const edm::Event&);
  void bookHistos(DQMStore *, std::vector<int>*, std::vector<int>*,std::vector<std::string>*,std::vector<std::string>*);
  //  void writeHistos();


 private:

  std::string dirname_;

  void setBits(std::vector<int>* l1bits, std::vector<int>* hltbits) {l1bits_=l1bits; hltbits_=hltbits;}
  double invariantMass(reco::Candidate*,reco::Candidate*);
  std::vector<int>* l1bits_;
  std::vector<int>* hltbits_;
    
    

  // Define the parameters
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
  
  int binFactor;

  reco::GsfElectronCollection theElectronCollection;
  reco::MuonCollection                  theMuonCollection    ;
  reco::PhotonCollection                thePhotonCollection  ;
  reco::CaloJetCollection               theCaloJetCollection ;
  reco::CaloMETCollection               theCaloMETCollection ;

  //histos

  //Jets
  MonitorElement* hJetMult;
  std::vector<MonitorElement*> hJetMultAfterL1;
  std::vector<MonitorElement*> hJetMultAfterHLT;
  MonitorElement* hJet1Pt;
  std::vector<MonitorElement*> hJet1PtAfterL1;
  std::vector<MonitorElement*> hJet1PtAfterHLT;
  MonitorElement* hJet2Pt;
  std::vector<MonitorElement*> hJet2PtAfterL1;
  std::vector<MonitorElement*> hJet2PtAfterHLT;
  MonitorElement* hJet1Eta;
  std::vector<MonitorElement*> hJet1EtaAfterL1;
  std::vector<MonitorElement*> hJet1EtaAfterHLT;
  MonitorElement* hJet2Eta;
  std::vector<MonitorElement*> hJet2EtaAfterL1;
  std::vector<MonitorElement*> hJet2EtaAfterHLT;
  MonitorElement* hJet1Phi;
  std::vector<MonitorElement*> hJet1PhiAfterL1;
  std::vector<MonitorElement*> hJet1PhiAfterHLT;
  MonitorElement* hJet2Phi;
  std::vector<MonitorElement*> hJet2PhiAfterL1;
  std::vector<MonitorElement*> hJet2PhiAfterHLT;

  MonitorElement* hDiJetInvMass;
  std::vector<MonitorElement*> hDiJetInvMassAfterL1;
  std::vector<MonitorElement*> hDiJetInvMassAfterHLT;




  //Electrons
  MonitorElement* hElecMult;
  std::vector<MonitorElement*> hElecMultAfterL1;
  std::vector<MonitorElement*> hElecMultAfterHLT;
  MonitorElement* hElec1Pt;
  std::vector<MonitorElement*> hElec1PtAfterL1;
  std::vector<MonitorElement*> hElec1PtAfterHLT;
  MonitorElement* hElec2Pt;
  std::vector<MonitorElement*> hElec2PtAfterL1;
  std::vector<MonitorElement*> hElec2PtAfterHLT;
  MonitorElement* hElec1Eta;
  std::vector<MonitorElement*> hElec1EtaAfterL1;
  std::vector<MonitorElement*> hElec1EtaAfterHLT;
  MonitorElement* hElec2Eta;
  std::vector<MonitorElement*> hElec2EtaAfterL1;
  std::vector<MonitorElement*> hElec2EtaAfterHLT;
  MonitorElement* hElec1Phi;
  std::vector<MonitorElement*> hElec1PhiAfterL1;
  std::vector<MonitorElement*> hElec1PhiAfterHLT;
  MonitorElement* hElec2Phi;
  std::vector<MonitorElement*> hElec2PhiAfterL1;
  std::vector<MonitorElement*> hElec2PhiAfterHLT;
  
  MonitorElement* hDiElecInvMass;
  std::vector<MonitorElement*> hDiElecInvMassAfterL1;
  std::vector<MonitorElement*> hDiElecInvMassAfterHLT;


  //Muons
  MonitorElement* hMuonMult;
  std::vector<MonitorElement*> hMuonMultAfterL1;
  std::vector<MonitorElement*> hMuonMultAfterHLT;
  MonitorElement* hMuon1Pt;
  std::vector<MonitorElement*> hMuon1PtAfterL1;
  std::vector<MonitorElement*> hMuon1PtAfterHLT;
  MonitorElement* hMuon2Pt;
  std::vector<MonitorElement*> hMuon2PtAfterL1;
  std::vector<MonitorElement*> hMuon2PtAfterHLT;
  MonitorElement* hMuon1Eta;
  std::vector<MonitorElement*> hMuon1EtaAfterL1;
  std::vector<MonitorElement*> hMuon1EtaAfterHLT;
  MonitorElement* hMuon2Eta;
  std::vector<MonitorElement*> hMuon2EtaAfterL1;
  std::vector<MonitorElement*> hMuon2EtaAfterHLT;
  MonitorElement* hMuon1Phi;
  std::vector<MonitorElement*> hMuon1PhiAfterL1;
  std::vector<MonitorElement*> hMuon1PhiAfterHLT;
  MonitorElement* hMuon2Phi;
  std::vector<MonitorElement*> hMuon2PhiAfterL1;
  std::vector<MonitorElement*> hMuon2PhiAfterHLT;

  MonitorElement* hDiMuonInvMass;
  std::vector<MonitorElement*> hDiMuonInvMassAfterL1;
  std::vector<MonitorElement*> hDiMuonInvMassAfterHLT;


  //Photons
  MonitorElement* hPhotonMult;
  std::vector<MonitorElement*> hPhotonMultAfterL1;
  std::vector<MonitorElement*> hPhotonMultAfterHLT;
  MonitorElement* hPhoton1Pt;
  std::vector<MonitorElement*> hPhoton1PtAfterL1;
  std::vector<MonitorElement*> hPhoton1PtAfterHLT;
  MonitorElement* hPhoton2Pt;
  std::vector<MonitorElement*> hPhoton2PtAfterL1;
  std::vector<MonitorElement*> hPhoton2PtAfterHLT;
  MonitorElement* hPhoton1Eta;
  std::vector<MonitorElement*> hPhoton1EtaAfterL1;
  std::vector<MonitorElement*> hPhoton1EtaAfterHLT;
  MonitorElement* hPhoton2Eta;
  std::vector<MonitorElement*> hPhoton2EtaAfterL1;
  std::vector<MonitorElement*> hPhoton2EtaAfterHLT;
  MonitorElement* hPhoton1Phi;
  std::vector<MonitorElement*> hPhoton1PhiAfterL1;
  std::vector<MonitorElement*> hPhoton1PhiAfterHLT;
  MonitorElement* hPhoton2Phi;
  std::vector<MonitorElement*> hPhoton2PhiAfterL1;
  std::vector<MonitorElement*> hPhoton2PhiAfterHLT;
  
  MonitorElement* hDiPhotonInvMass;
  std::vector<MonitorElement*> hDiPhotonInvMassAfterL1;
  std::vector<MonitorElement*> hDiPhotonInvMassAfterHLT;

  
  //MET
  MonitorElement* hMET;
  MonitorElement* hMETphi;
  MonitorElement* hMETx;
  MonitorElement* hMETy;
  MonitorElement* hSumEt;
  MonitorElement* hMETSignificance;

  std::vector<MonitorElement*> hMETAfterL1;
  std::vector<MonitorElement*> hMETAfterHLT;
  std::vector<MonitorElement*> hMETphiAfterL1;
  std::vector<MonitorElement*> hMETphiAfterHLT;
  std::vector<MonitorElement*> hMETxAfterL1;
  std::vector<MonitorElement*> hMETxAfterHLT;
  std::vector<MonitorElement*> hMETyAfterL1;
  std::vector<MonitorElement*> hMETyAfterHLT;
  std::vector<MonitorElement*> hSumEtAfterL1;
  std::vector<MonitorElement*> hSumEtAfterHLT;
  std::vector<MonitorElement*> hMETSignificanceAfterL1;
  std::vector<MonitorElement*> hMETSignificanceAfterHLT;





  std::string myHistoName;
  std::string myHistoTitle;

};


#endif
