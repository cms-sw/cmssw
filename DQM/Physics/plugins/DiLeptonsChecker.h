// system include files
#include <memory>
#include <string>
#include <vector>
#include "TLorentzVector.h"

// FWCore
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
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
#include "RecoEgamma/Examples/plugins/ElectronIDAnalyzer.h"
#include "DataFormats/Common/interface/ValueMap.h"

//user files
#include "DQM/Physics/interface/Selection.h"
#include "DQM/Physics/interface/MuonChecker.h"
#include "DQM/Physics/interface/ElectronChecker.h"
#include "DQM/Physics/interface/MetChecker.h"
#include "DQM/Physics/interface/JetChecker.h"
#include "DQM/Physics/interface/KinematicsChecker.h"
//jet corrections
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

/**
   \class DiLeptonsChecker DiLeptonsChecker.cc DQM/Physics/plugins/DiLeptonsChecker.h

   \brief   Add a one sentence description here...

   Module dedicated to dilepton channel
   It takes an edm::View of objects as inputs:
   Muons, CaloJets, CaloMETs, trigger
   It's an EDAnalyzer
   It uses DQMStore to store the histograms
   in a directory: "DiLeptonsJetChecker"
   This module will call the following modules:
   MuonChecker, JetChecker, METChecker, KinematicsChecker
   taken as input the selected objects (according to the configuration)
   Still missing !!! :
   - add b-tagging
*/

class DiLeptonsChecker : public edm::EDAnalyzer {
public:
  explicit DiLeptonsChecker(const edm::ParameterSet&);
  ~DiLeptonsChecker();
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  double ComputeNbEvent(MonitorElement* h, int bin);
  double ComputeNbEventError(MonitorElement* h, int bin);
  
  bool verbose_;
  DQMStore* dqmStore_;
  std::string outputFileName_;
  
  edm::InputTag labelMuons_;
  edm::InputTag labelElectrons_;
  edm::InputTag labelJets_;
  edm::InputTag labelMETs_;
  edm::InputTag labelBeamSpot_;
  edm::InputTag labelTriggerResults_;
  std::string electronIDLabel_ ;
  
  bool lookAtDiElectronsChannel_ ;     
  bool lookAtDiMuonsChannel_ ;          
  bool lookAtElectronMuonChannel_ ;   
  bool useJES_; 

  //Common tools:modules
  //no selection
  JetChecker*        jetChecker;
  MetChecker*        metChecker;
  MuonChecker*       muonChecker;
  ElectronChecker*   electronChecker;
  KinematicsChecker* kinematicsChecker;
  
  //trigger sel.
  JetChecker*        jetCheckerTrigger;
  MetChecker*        metCheckerTrigger;
  MuonChecker*       muonCheckerTrigger;
  ElectronChecker*   electronCheckerTrigger;
  KinematicsChecker* kinematicsCheckerTrigger;
  
  //leptons non iso selection
  JetChecker*        jetCheckerNonIsoLept;
  MetChecker*        metCheckerNonIsoLept;
  MuonChecker*       muonCheckerNonIsoLept;
  ElectronChecker*   electronCheckerNonIsoLept;
  KinematicsChecker* kinematicsCheckerNonIsoLept;
  
  //leptons selection
  JetChecker*        jetCheckerIsoLept;
  MetChecker*        metCheckerIsoLept;
  MuonChecker*       muonCheckerIsoLept;
  ElectronChecker*   electronCheckerIsoLept;
  KinematicsChecker* kinematicsCheckerIsoLept;
  
  //leptons pair selection
  JetChecker*        jetCheckerLeptPair;
  MetChecker*        metCheckerLeptPair;
  MuonChecker*       muonCheckerLeptPair;
  ElectronChecker*   electronCheckerLeptPair;
  KinematicsChecker* kinematicsCheckerLeptPair;
  
  //Inv mass pair selection
  JetChecker*        jetCheckerInvM;
  MetChecker*        metCheckerInvM;
  MuonChecker*       muonCheckerInvM;
  ElectronChecker*   electronCheckerInvM;
  KinematicsChecker* kinematicsCheckerInvM;
  
  //Jets selection
  JetChecker*        jetCheckerJet;
  MetChecker*        metCheckerJet;
  MuonChecker*       muonCheckerJet;
  ElectronChecker*   electronCheckerJet;
  KinematicsChecker* kinematicsCheckerJet;
  
  //met selection
  JetChecker*        jetCheckerMet;
  MetChecker*        metCheckerMet;
  MuonChecker*       muonCheckerMet;
  ElectronChecker*   electronCheckerMet;
  KinematicsChecker* kinematicsCheckerMet;
  
  //btag selection
  JetChecker*        jetCheckerBtag;
  MetChecker*        metCheckerBtag;
  MuonChecker*       muonCheckerBtag;
  ElectronChecker*   electronCheckerBtag;
  KinematicsChecker* kinematicsCheckerBtag;
  
  //double btag selection
  JetChecker*        jetCheckerDBtag;
  MetChecker*        metCheckerDBtag;
  MuonChecker*       muonCheckerDBtag;
  ElectronChecker*   electronCheckerDBtag;
  KinematicsChecker* kinematicsCheckerDBtag;
  
  //Can declare many time the same module and use them for different selection

  //Configuration
  int    nofJets_;
  double ptThrJets_;
  double etaThrJets_;
  double eHThrJets_;
  double ptThrMuons_;
  double etaThrMuons_;
  double muonRelIso_;
  double muonRelIsoCalo_;
  double muonRelIsoTrk_;
  double muonVetoEM_;
  double muonVetoHad_;
  double muonD0Cut_;
  double electronD0Cut_;
  double ptThrElectrons_;
  double etaThrElectrons_;
  double electronRelIso_;
  double electronRelIsoCalo_;
  double electronRelIsoTrk_;
  double metCut_;
  int    chi2Cut_;
  int    nofValidHits_;
  double  deltaREMCut_;
  
  std::vector<std::string> triggerPath_;
  int            luminosity_;
  double         xsection_;
  std::string    jetCorrector_;
 
  std::string relativePath_; // name of the directory where all the histograms will be stored
  //Histograms are booked in the beginJob() method
  std::map<std::string, MonitorElement*> histocontainer_;       // simple map to contain all TH1D.
};
