// system include files
#include <iostream>
#include <memory>
#include <string>
#include <vector>
// FWCore
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
//needed for MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//DataFormat
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
//user files
#include "DQM/Physics/interface/Selection.h"
#include "DQM/Physics/interface/MuonChecker.h"
#include "DQM/Physics/interface/ElectronChecker.h"
#include "DQM/Physics/interface/MetChecker.h"
#include "DQM/Physics/interface/JetChecker.h"
#include "DQM/Physics/interface/KinematicsChecker.h"
#include "DQM/Physics/interface/SemiLeptonChecker.h"
// Root
#include <TString.h>

/**
   \class   LeptonJetsChecker LeptonJetsChecker.h "DQM/Physics/plugins/LeptonJetsChecker.h"

   \brief   Add a one sentence description here...

   Module dedicated to Lepton+jets channel
   It takes an edm::View of objects as inputs:
   Muons, CaloJets, CaloMETs, trigger
   It's an EDAnalyzer
   It uses DQMStore to store the histograms
   in a directory: "LeptonJetChecker"
   This module will call the following modules:
   MuonChecker, JetChecker, METChecker, KinematicsChecker, SemiLeotpnChecker
   taken as input the selected objects (according to the configuration)
   Still missing !!! :
   - add relevant plots combining objects: angles, Kinematics (Pt(top)) ..
   - update Selection Table if needed
*/

class LeptonJetsChecker : public edm::EDAnalyzer {
public:
  explicit LeptonJetsChecker(const edm::ParameterSet&);
  ~LeptonJetsChecker();
  
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  double ComputeNbEvent(MonitorElement* h, int bin);
  
  DQMStore* dqmStore_;
  std::string outputFileName_;
  bool saveDQMMEs_ ;
  
  edm::InputTag labelMuons_;
  edm::InputTag labelElectrons_;
  edm::InputTag labelJets_;
  edm::InputTag labelMETs_;
  edm::InputTag labelBeamSpot_;
  edm::InputTag labelTriggerResults_;
  
  //Common tools:modules
  //before selection
  JetChecker*        jetCheckerNoSel;
  MetChecker*        metCheckerNoSel;
  MuonChecker*       muonCheckerNoSel;
  ElectronChecker*   electronCheckerNoSel;
  KinematicsChecker* kinematicsCheckerNoSel;
  
  JetChecker*        jetCheckerLeptonNonIso;
  MetChecker*        metCheckerLeptonNonIso;
  MuonChecker*       muonCheckerLeptonNonIso;
  ElectronChecker*   electronCheckerLeptonNonIso;
  KinematicsChecker* kinematicsCheckerLeptonNonIso;
  
  JetChecker*        jetCheckerLeptonIso;
  MetChecker*        metCheckerLeptonIso;
  MuonChecker*       muonCheckerLeptonIso;
  ElectronChecker*   electronCheckerLeptonIso;
  KinematicsChecker* kinematicsCheckerLeptonIso;
  
  JetChecker*        jetCheckerVetoOtherLeptonType;
  MetChecker*        metCheckerVetoOtherLeptonType;
  MuonChecker*       muonCheckerVetoOtherLeptonType;
  ElectronChecker*   electronCheckerVetoOtherLeptonType;
  KinematicsChecker* kinematicsCheckerVetoOtherLeptonType;
  
  JetChecker*        jetCheckerVetoLooseMuon;
  MetChecker*        metCheckerVetoLooseMuon;
  MuonChecker*       muonCheckerVetoLooseMuon;
  ElectronChecker*   electronCheckerVetoLooseMuon;
  KinematicsChecker* kinematicsCheckerVetoLooseMuon;
  
  JetChecker*        jetCheckerVetoLooseElectron;
  MetChecker*        metCheckerVetoLooseElectron;
  MuonChecker*       muonCheckerVetoLooseElectron;
  ElectronChecker*   electronCheckerVetoLooseElectron;
  KinematicsChecker* kinematicsCheckerVetoLooseElectron;
  
  JetChecker*        jetCheckerMET;
  MetChecker*        metCheckerMET;
  MuonChecker*       muonCheckerMET;
  ElectronChecker*   electronCheckerMET;
  KinematicsChecker* kinematicsCheckerMET;
  
  JetChecker*        jetChecker1Jets;
  MetChecker*        metChecker1Jets;
  MuonChecker*       muonChecker1Jets;
  ElectronChecker*   electronChecker1Jets;
  KinematicsChecker* kinematicsChecker1Jets;
  
  JetChecker*        jetChecker2Jets;
  MetChecker*        metChecker2Jets;
  MuonChecker*       muonChecker2Jets;
  ElectronChecker*   electronChecker2Jets;
  KinematicsChecker* kinematicsChecker2Jets;
  
  JetChecker*        jetChecker3Jets;
  MetChecker*        metChecker3Jets;
  MuonChecker*       muonChecker3Jets;
  ElectronChecker*   electronChecker3Jets;
  KinematicsChecker* kinematicsChecker3Jets;
  
  JetChecker*        jetChecker4Jets;
  MetChecker*        metChecker4Jets;
  MuonChecker*       muonChecker4Jets;
  ElectronChecker*   electronChecker4Jets;
  KinematicsChecker* kinematicsChecker4Jets;
  
  JetChecker*        jetChecker;
  MetChecker*        metChecker;
  MuonChecker*       muonChecker;
  ElectronChecker*   electronChecker;
  KinematicsChecker* kinematicsChecker;
  
  SemiLeptonChecker* semiLeptonChecker;
  
  //Checkers for October exercise
  JetChecker*        jetCheckerDeltaR;
  MetChecker*        metCheckerDeltaR;
  MuonChecker*       muonCheckerDeltaR;
  ElectronChecker*   electronCheckerDeltaR;
  KinematicsChecker* kinematicsCheckerDeltaR;
  
  //Configuration variables
  //Jets
  int NofJets;
  double PtThrJets;
  double EtaThrJets;
  double EHThrJets;
  double JetDeltaRLeptonJetThreshold;
  bool   applyLeptonJetDeltaRCut;
  //Muons
  double PtThrMuons;
  double EtaThrMuons;
  double MuonRelIso;
  double MuonVetoEM;
  double MuonVetoHad;
  double MuonD0Cut;
  int Chi2Cut;
  int NofValidHits;
  //Electrons
  bool useElectronID_;
  std::string electronIDLabel;
  double PtThrElectrons;
  double EtaThrElectrons;  
  double ElectronRelIso;
  double ElectronD0Cut;
  bool vetoEBEETransitionRegion;
  
  double METThreshold;
  bool ApplyMETCut_;
  std::string triggerPath;
  bool useTrigger_;
  int Luminosity;
  double Xsection;
  //Loose leptons
  bool VetoLooseLepton_;
  double PtThrMuonLoose;           
  double EtaThrMuonLoose;          
  double RelIsoThrMuonLoose;      
  double PtThrElectronLoose;      
  double EtaThrElectronLoose;      
  double RelIsoThrElectronLoose;   
  std::string electronIDLabelLoose;
  
  std::string   jetCorrector_; 
  bool useJES_; 
  
  std::string leptonType_;
  std::string otherLeptonType_;
  
  bool PerformOctoberXDeltaRStep_; 
  
  // name of the directory where all the histograms will be stored
  std::string relativePath_;
  //Histogram container
  std::map<std::string,MonitorElement*> histocontainer_;
};
