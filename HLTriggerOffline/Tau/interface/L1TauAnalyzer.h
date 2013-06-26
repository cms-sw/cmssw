// -*- C++ -*-
//
// Package:    L1TauAnalyzer
// Class:      L1TauAnalyzer
// 
/**\class L1TauAnalyzer L1TauAnalyzer.cc UserCode/L1TauAnalyzer/src/L1TauAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chi Nhan Nguyen
//         Created:  Fri Feb 22 09:20:55 CST 2008
// $Id: L1TauAnalyzer.h,v 1.4 2010/01/12 06:38:12 hegner Exp $
//
//


// system include files
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminatorByIsolation.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <Math/GenVector/VectorUtil.h>
#include "TLorentzVector.h"

#include "FWCore/ServiceRegistry/interface/Service.h" // Framework services
#include "CommonTools/UtilAlgos/interface/TFileService.h" // Framework service for histograms

#include "TH1.h"

// L1 Trigger data formats
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"

//
// class decleration
//

class L1TauAnalyzer : public edm::EDAnalyzer {
   public:
      explicit L1TauAnalyzer(const edm::ParameterSet&);
      ~L1TauAnalyzer();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;


  void getGenObjects(const edm::Event&, const edm::EventSetup&);
  void getPFTauObjects(const edm::Event&, const edm::EventSetup&);
  void getL1extraObjects(const edm::Event&, const edm::EventSetup&);

  void fillL1Histograms();
  void fillGenHistograms();
  void fillPFTauHistograms();

  void evalL1Decisions(const edm::Event& iEvent);
  void evalL1extraDecisions();

  void calcL1MCTauMatching();
  void calcL1MCPFTauMatching();

  void convertToIntegratedEff(TH1*,double);
  void printTrigReport();
  
  // ----------member data ---------------------------

  edm::InputTag _PFTauSource;
  edm::InputTag _PFTauDiscriminatorSource;
  edm::InputTag _GenParticleSource;

  edm::InputTag _L1extraTauJetSource;
  edm::InputTag _L1extraCenJetSource;
  edm::InputTag _L1extraForJetSource;
  edm::InputTag _L1extraNonIsoEgammaSource;
  edm::InputTag _L1extraIsoEgammaSource;
  edm::InputTag _L1extraMETSource;
  edm::InputTag _L1extraMuonSource;

  edm::InputTag _L1GtReadoutRecord;
  edm::InputTag _L1GtObjectMap;

  bool _DoMCMatching;
  bool _DoPFTauMatching;

  // PDG id for 
  int _BosonPID;

  // Cuts
  double _L1MCTauMinDeltaR;
  double _MCTauHadMinEt;
  double _MCTauHadMaxAbsEta;

  double _PFMCTauMinDeltaR;
  double _PFTauMinEt;
  double _PFTauMaxAbsEta;

  // Thresholds of L1 menu
  double _SingleTauThreshold;
  double _DoubleTauThreshold;
  std::vector<double> _SingleTauMETThresholds;
  std::vector<double> _MuTauThresholds;
  std::vector<double> _IsoEgTauThresholds;

  // L1 menu names
  std::string _L1SingleTauName;
  std::string _L1DoubleTauName;
  std::string _L1TauMETName;
  std::string _L1MuonTauName;
  std::string _L1IsoEgTauName;

  // Gen Objects
  std::vector<TLorentzVector> _GenTauMuons;
  std::vector<TLorentzVector> _GenTauElecs;
  std::vector<TLorentzVector> _GenTauHads;

  // Tagged PFTau Objects
  std::vector<TLorentzVector> _PFTaus;

  // L1extra Objects
  std::vector<TLorentzVector> _L1Taus;
  std::vector<TLorentzVector> _L1CenJets;
  std::vector<TLorentzVector> _L1ForJets;
  std::vector<TLorentzVector> _L1NonIsoEgammas;
  std::vector<TLorentzVector> _L1IsoEgammas;
  std::vector<TLorentzVector> _L1METs;
  std::vector<TLorentzVector> _L1Muons;
  std::vector<int> _L1MuQuals;


  // histograms
  TH1* h_L1TauEt;
  TH1* h_L1TauEta;
  TH1* h_L1TauPhi;

  TH1* h_L1Tau1Et;
  TH1* h_L1Tau1Eta;
  TH1* h_L1Tau1Phi;

  TH1* h_L1Tau2Et;
  TH1* h_L1Tau2Eta;
  TH1* h_L1Tau2Phi;

  //
  TH1* h_GenTauHadEt;
  TH1* h_GenTauHadEta;
  TH1* h_GenTauHadPhi;

  //
  TH1* h_PFTauEt;
  TH1* h_PFTauEta;
  TH1* h_PFTauPhi;

  // L1 response
  TH1* h_L1MCTauDeltaR;
  TH1* h_L1minusMCTauEt;
  TH1* h_L1minusMCoverMCTauEt;

  // MC matching efficiencies
  TH1* h_EffMCTauEt;
  TH1* h_EffMCTauEta;
  TH1* h_EffMCTauPhi;
  // Numerators
  TH1* h_L1MCMatchedTauEt;
  TH1* h_L1MCMatchedTauEta;
  TH1* h_L1MCMatchedTauPhi;
  // Denominators
  TH1* h_MCTauHadEt;
  TH1* h_MCTauHadEta;
  TH1* h_MCTauHadPhi;

  // MCPF matching efficiencies
  TH1* h_EffMCPFTauEt;
  TH1* h_EffMCPFTauEta;
  TH1* h_EffMCPFTauPhi;
  // Numerators
  TH1* h_L1MCPFMatchedTauEt;
  TH1* h_L1MCPFMatchedTauEta;
  TH1* h_L1MCPFMatchedTauPhi;
  // Denominators
  TH1* h_MCPFTauHadEt;
  TH1* h_MCPFTauHadEta;
  TH1* h_MCPFTauHadPhi;

  TH1* h_PFMCTauDeltaR;

  
  // Event based efficiencies as a function of thresholds
  TH1* h_L1SingleTauEffEt;
  TH1* h_L1DoubleTauEffEt;
  TH1* h_L1SingleTauEffMCMatchEt;
  TH1* h_L1DoubleTauEffMCMatchEt;
  TH1* h_L1SingleTauEffPFMCMatchEt;
  TH1* h_L1DoubleTauEffPFMCMatchEt;

// Counters for event based efficiencies
  int _nEvents; // all events processed

  int _nEventsGenTauHad; 
  int _nEventsDoubleGenTauHads; 
  int _nEventsGenTauMuonTauHad; 
  int _nEventsGenTauElecTauHad;   

  int _nfidEventsGenTauHad; 
  int _nfidEventsDoubleGenTauHads; 
  int _nfidEventsGenTauMuonTauHad; 
  int _nfidEventsGenTauElecTauHad;   

  int _nEventsPFMatchGenTauHad; 
  int _nEventsPFMatchDoubleGenTauHads; 
  int _nEventsPFMatchGenTauMuonTauHad; 
  int _nEventsPFMatchGenTauElecTauHad;   

  int _nEventsL1SingleTauPassed;
  int _nEventsL1SingleTauPassedMCMatched;
  int _nEventsL1SingleTauPassedPFMCMatched;

  int _nEventsL1DoubleTauPassed;
  int _nEventsL1DoubleTauPassedMCMatched;
  int _nEventsL1DoubleTauPassedPFMCMatched;

  int _nEventsL1SingleTauMETPassed;
  int _nEventsL1SingleTauMETPassedMCMatched;
  int _nEventsL1SingleTauMETPassedPFMCMatched;

  int _nEventsL1MuonTauPassed;
  int _nEventsL1MuonTauPassedMCMatched;
  int _nEventsL1MuonTauPassedPFMCMatched;

  int _nEventsL1IsoEgTauPassed;
  int _nEventsL1IsoEgTauPassedMCMatched;
  int _nEventsL1IsoEgTauPassedPFMCMatched;

  // from GT bit info
  int _nEventsL1GTSingleTauPassed;
  int _nEventsL1GTDoubleTauPassed;
  int _nEventsL1GTSingleTauMETPassed;
  int _nEventsL1GTMuonTauPassed;
  int _nEventsL1GTIsoEgTauPassed;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

