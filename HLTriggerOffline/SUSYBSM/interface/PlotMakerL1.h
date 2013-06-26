#ifndef PlotMakerL1_h
#define PlotMakerL1_h

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



class PlotMakerL1 {

 public:
  PlotMakerL1(edm::ParameterSet objectList);
  virtual ~PlotMakerL1(){};

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
  std::string m_l1extra;

  l1extra::L1EmParticleCollection theL1EmIsoCollection, theL1EmNotIsoCollection;
  l1extra::L1MuonParticleCollection theL1MuonCollection;
  l1extra::L1JetParticleCollection theL1CentralJetCollection, theL1ForwardJetCollection, theL1TauJetCollection;
  l1extra::L1EtMissParticleCollection theL1METCollection;

  //histos

  //Jets
  MonitorElement* hL1CentralJetMult;
  MonitorElement* hL1ForwardJetMult;
  MonitorElement* hL1TauJetMult;
  std::vector<MonitorElement*> hL1CentralJetMultAfterL1;
  std::vector<MonitorElement*> hL1CentralJetMultAfterHLT;
  std::vector<MonitorElement*> hL1ForwardJetMultAfterL1;
  std::vector<MonitorElement*> hL1ForwardJetMultAfterHLT;
  std::vector<MonitorElement*> hL1TauJetMultAfterL1;
  std::vector<MonitorElement*> hL1TauJetMultAfterHLT;
  MonitorElement* hL1CentralJet1Pt;
  MonitorElement* hL1ForwardJet1Pt;
  MonitorElement* hL1TauJet1Pt;
  std::vector<MonitorElement*> hL1CentralJet1PtAfterL1;
  std::vector<MonitorElement*> hL1CentralJet1PtAfterHLT;
  std::vector<MonitorElement*> hL1ForwardJet1PtAfterL1;
  std::vector<MonitorElement*> hL1ForwardJet1PtAfterHLT;
  std::vector<MonitorElement*> hL1TauJet1PtAfterL1;
  std::vector<MonitorElement*> hL1TauJet1PtAfterHLT;
  MonitorElement* hL1CentralJet2Pt;
  MonitorElement* hL1ForwardJet2Pt;
  MonitorElement* hL1TauJet2Pt;
  std::vector<MonitorElement*> hL1CentralJet2PtAfterL1;
  std::vector<MonitorElement*> hL1CentralJet2PtAfterHLT;
  std::vector<MonitorElement*> hL1ForwardJet2PtAfterL1;
  std::vector<MonitorElement*> hL1ForwardJet2PtAfterHLT;
  std::vector<MonitorElement*> hL1TauJet2PtAfterL1;
  std::vector<MonitorElement*> hL1TauJet2PtAfterHLT;
  MonitorElement* hL1CentralJet1Eta;
  MonitorElement* hL1ForwardJet1Eta;
  MonitorElement* hL1TauJet1Eta;
  std::vector<MonitorElement*> hL1CentralJet1EtaAfterL1;
  std::vector<MonitorElement*> hL1CentralJet1EtaAfterHLT;
  std::vector<MonitorElement*> hL1ForwardJet1EtaAfterL1;
  std::vector<MonitorElement*> hL1ForwardJet1EtaAfterHLT;
  std::vector<MonitorElement*> hL1TauJet1EtaAfterL1;
  std::vector<MonitorElement*> hL1TauJet1EtaAfterHLT;
  MonitorElement* hL1CentralJet2Eta;
  MonitorElement* hL1ForwardJet2Eta;
  MonitorElement* hL1TauJet2Eta;
  std::vector<MonitorElement*> hL1CentralJet2EtaAfterL1;
  std::vector<MonitorElement*> hL1CentralJet2EtaAfterHLT;
  std::vector<MonitorElement*> hL1ForwardJet2EtaAfterL1;
  std::vector<MonitorElement*> hL1ForwardJet2EtaAfterHLT;
  std::vector<MonitorElement*> hL1TauJet2EtaAfterL1;
  std::vector<MonitorElement*> hL1TauJet2EtaAfterHLT;
  MonitorElement* hL1CentralJet1Phi;
  MonitorElement* hL1ForwardJet1Phi;
  MonitorElement* hL1TauJet1Phi;
  std::vector<MonitorElement*> hL1CentralJet1PhiAfterL1;
  std::vector<MonitorElement*> hL1CentralJet1PhiAfterHLT;
  std::vector<MonitorElement*> hL1ForwardJet1PhiAfterL1;
  std::vector<MonitorElement*> hL1ForwardJet1PhiAfterHLT;
  std::vector<MonitorElement*> hL1TauJet1PhiAfterL1;
  std::vector<MonitorElement*> hL1TauJet1PhiAfterHLT;
  MonitorElement* hL1CentralJet2Phi;
  MonitorElement* hL1ForwardJet2Phi;
  MonitorElement* hL1TauJet2Phi;
  std::vector<MonitorElement*> hL1CentralJet2PhiAfterL1;
  std::vector<MonitorElement*> hL1CentralJet2PhiAfterHLT;
  std::vector<MonitorElement*> hL1ForwardJet2PhiAfterL1;
  std::vector<MonitorElement*> hL1ForwardJet2PhiAfterHLT;
  std::vector<MonitorElement*> hL1TauJet2PhiAfterL1;
  std::vector<MonitorElement*> hL1TauJet2PhiAfterHLT;



  //Electrons
  MonitorElement* hL1EmIsoMult;
  MonitorElement* hL1EmNotIsoMult;
  std::vector<MonitorElement*> hL1EmIsoMultAfterL1;
  std::vector<MonitorElement*> hL1EmIsoMultAfterHLT;
  std::vector<MonitorElement*> hL1EmNotIsoMultAfterL1;
  std::vector<MonitorElement*> hL1EmNotIsoMultAfterHLT;
  MonitorElement* hL1EmIso1Pt;
  MonitorElement* hL1EmNotIso1Pt;
  std::vector<MonitorElement*> hL1EmIso1PtAfterL1;
  std::vector<MonitorElement*> hL1EmIso1PtAfterHLT;
  std::vector<MonitorElement*> hL1EmNotIso1PtAfterL1;
  std::vector<MonitorElement*> hL1EmNotIso1PtAfterHLT;
  MonitorElement* hL1EmIso2Pt;
  MonitorElement* hL1EmNotIso2Pt;
  std::vector<MonitorElement*> hL1EmIso2PtAfterL1;
  std::vector<MonitorElement*> hL1EmIso2PtAfterHLT;
  std::vector<MonitorElement*> hL1EmNotIso2PtAfterL1;
  std::vector<MonitorElement*> hL1EmNotIso2PtAfterHLT;
  MonitorElement* hL1EmIso1Eta;
  MonitorElement* hL1EmNotIso1Eta;
  std::vector<MonitorElement*> hL1EmIso1EtaAfterL1;
  std::vector<MonitorElement*> hL1EmIso1EtaAfterHLT;
  std::vector<MonitorElement*> hL1EmNotIso1EtaAfterL1;
  std::vector<MonitorElement*> hL1EmNotIso1EtaAfterHLT;
  MonitorElement* hL1EmIso2Eta;
  MonitorElement* hL1EmNotIso2Eta;
  std::vector<MonitorElement*> hL1EmIso2EtaAfterL1;
  std::vector<MonitorElement*> hL1EmIso2EtaAfterHLT;
  std::vector<MonitorElement*> hL1EmNotIso2EtaAfterL1;
  std::vector<MonitorElement*> hL1EmNotIso2EtaAfterHLT;
  MonitorElement* hL1EmIso1Phi;
  MonitorElement* hL1EmNotIso1Phi;
  std::vector<MonitorElement*> hL1EmIso1PhiAfterL1;
  std::vector<MonitorElement*> hL1EmIso1PhiAfterHLT;
  std::vector<MonitorElement*> hL1EmNotIso1PhiAfterL1;
  std::vector<MonitorElement*> hL1EmNotIso1PhiAfterHLT;
  MonitorElement* hL1EmIso2Phi;
  MonitorElement* hL1EmNotIso2Phi;
  std::vector<MonitorElement*> hL1EmIso2PhiAfterL1;
  std::vector<MonitorElement*> hL1EmIso2PhiAfterHLT;
  std::vector<MonitorElement*> hL1EmNotIso2PhiAfterL1;
  std::vector<MonitorElement*> hL1EmNotIso2PhiAfterHLT;
  

  //Muons
  MonitorElement* hL1MuonMult;
  std::vector<MonitorElement*> hL1MuonMultAfterL1;
  std::vector<MonitorElement*> hL1MuonMultAfterHLT;
  MonitorElement* hL1Muon1Pt;
  std::vector<MonitorElement*> hL1Muon1PtAfterL1;
  std::vector<MonitorElement*> hL1Muon1PtAfterHLT;
  MonitorElement* hL1Muon2Pt;
  std::vector<MonitorElement*> hL1Muon2PtAfterL1;
  std::vector<MonitorElement*> hL1Muon2PtAfterHLT;
  MonitorElement* hL1Muon1Eta;
  std::vector<MonitorElement*> hL1Muon1EtaAfterL1;
  std::vector<MonitorElement*> hL1Muon1EtaAfterHLT;
  MonitorElement* hL1Muon2Eta;
  std::vector<MonitorElement*> hL1Muon2EtaAfterL1;
  std::vector<MonitorElement*> hL1Muon2EtaAfterHLT;
  MonitorElement* hL1Muon1Phi;
  std::vector<MonitorElement*> hL1Muon1PhiAfterL1;
  std::vector<MonitorElement*> hL1Muon1PhiAfterHLT;
  MonitorElement* hL1Muon2Phi;
  std::vector<MonitorElement*> hL1Muon2PhiAfterL1;
  std::vector<MonitorElement*> hL1Muon2PhiAfterHLT;


  //MET
  MonitorElement* hL1MET;
  MonitorElement* hL1METphi;
  MonitorElement* hL1METx;
  MonitorElement* hL1METy;
  MonitorElement* hL1SumEt;
  MonitorElement* hL1METSignificance;

  std::vector<MonitorElement*> hL1METAfterL1;
  std::vector<MonitorElement*> hL1METAfterHLT;
  std::vector<MonitorElement*> hL1METphiAfterL1;
  std::vector<MonitorElement*> hL1METphiAfterHLT;
  std::vector<MonitorElement*> hL1METxAfterL1;
  std::vector<MonitorElement*> hL1METxAfterHLT;
  std::vector<MonitorElement*> hL1METyAfterL1;
  std::vector<MonitorElement*> hL1METyAfterHLT;
  std::vector<MonitorElement*> hL1SumEtAfterL1;
  std::vector<MonitorElement*> hL1SumEtAfterHLT;
  std::vector<MonitorElement*> hL1METSignificanceAfterL1;
  std::vector<MonitorElement*> hL1METSignificanceAfterHLT;

  std::string myHistoName;
  std::string myHistoTitle;

};


#endif
