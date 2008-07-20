// Original Author:  Chi Nhan Nguyen
//         Created:  Fri Feb 22 09:20:55 CST 2008

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

#include "FWCore/ServiceRegistry/interface/Service.h" // Framework services
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//
typedef math::XYZTLorentzVectorD   LV;
typedef std::vector<LV>            LVColl;


class L1TauValidation : public edm::EDAnalyzer {
   public:
      explicit L1TauValidation(const edm::ParameterSet&);
      ~L1TauValidation();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;


  void getL1extraObjects(const edm::Event&);
  void evalL1extraDecisions();
  void evalL1Decisions(const edm::Event& iEvent);

  void fillL1Histograms();
  void fillL1MCTauMatchedHists(const edm::Event& iEvent);

  void convertToIntegratedEff(MonitorElement*,double);
  
  // ----------member data ---------------------------

  edm::InputTag     _mcColl;         // input products from HLTMcInfo
  
  edm::InputTag _L1extraTauJetSource;
  edm::InputTag _L1extraCenJetSource;
  edm::InputTag _L1extraForJetSource;
  edm::InputTag _L1extraMuonSource;
  edm::InputTag _L1extraMETSource;
  edm::InputTag _L1extraNonIsoEgammaSource;
  edm::InputTag _L1extraIsoEgammaSource;

  edm::InputTag _L1GtReadoutRecord;
  edm::InputTag _L1GtObjectMap;

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

  // Cuts
  double _L1MCTauMinDeltaR;
  double _MCTauHadMinEt;
  double _MCTauHadMaxAbsEta;

  //Output file
  std::string _triggerTag;//tag for dqm flder
  std::string _outFile;

  // L1extra Objects
  LVColl _L1Taus;
  LVColl _L1CenJets;
  LVColl _L1ForJets;
  LVColl _L1NonIsoEgammas;
  LVColl _L1IsoEgammas;
  LVColl _L1METs;
  LVColl _L1Muons;
  std::vector<int> _L1MuQuals;
  
  // histograms
  MonitorElement* h_L1TauEt;  
  MonitorElement* h_L1TauEta;
  MonitorElement* h_L1TauPhi;

  MonitorElement* h_L1Tau1Et;
  MonitorElement* h_L1Tau1Eta;
  MonitorElement* h_L1Tau1Phi;

  MonitorElement* h_L1Tau2Et;
  MonitorElement* h_L1Tau2Eta;
  MonitorElement* h_L1Tau2Phi;

  // L1 response
  MonitorElement* h_L1MCTauDeltaR;
  MonitorElement* h_L1minusMCTauEt;
  MonitorElement* h_L1minusMCoverMCTauEt;

  // MC w/o cuts
  MonitorElement* h_GenTauHadEt;
  MonitorElement* h_GenTauHadEta;
  MonitorElement* h_GenTauHadPhi;

  // MC matching efficiencies
  MonitorElement* h_EffMCTauEt;
  MonitorElement* h_EffMCTauEta;
  MonitorElement* h_EffMCTauPhi;
  // Numerators
  MonitorElement* h_L1MCMatchedTauEt;
  MonitorElement* h_L1MCMatchedTauEta;
  MonitorElement* h_L1MCMatchedTauPhi;
  // Denominators
  MonitorElement* h_MCTauHadEt;
  MonitorElement* h_MCTauHadEta;
  MonitorElement* h_MCTauHadPhi;
  
  // Event based efficiencies as a function of thresholds
  MonitorElement* h_L1SingleTauEffEt;
  MonitorElement* h_L1DoubleTauEffEt;
  MonitorElement* h_L1SingleTauEffMCMatchEt;
  MonitorElement* h_L1DoubleTauEffMCMatchEt;

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

  int _nEventsL1DoubleTauPassed;
  int _nEventsL1DoubleTauPassedMCMatched;

  int _nEventsL1SingleTauMETPassed;
  int _nEventsL1SingleTauMETPassedMCMatched;

  int _nEventsL1MuonTauPassed;
  int _nEventsL1MuonTauPassedMCMatched;

  int _nEventsL1IsoEgTauPassed;
  int _nEventsL1IsoEgTauPassedMCMatched;

  // from GT bit info
  int _nEventsL1GTSingleTauPassed;
  int _nEventsL1GTDoubleTauPassed;
  int _nEventsL1GTSingleTauMETPassed;
  int _nEventsL1GTMuonTauPassed;
  int _nEventsL1GTIsoEgTauPassed;

};

//

