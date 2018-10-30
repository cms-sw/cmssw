// -*- C++ -*-
//
// Package:    QM/HLTObjectMonitorHI
// Class:      HLTObjectMonitorHI
//
/**\class HLTObjectMonitorHI HLTObjectMonitorHI.cc DQM/HLTEvF/plugins/HLTObjectMonitorHI.cc 

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Geonhee Oh
//         Created:  Sun, 29 Oct 2018 17:00:00 GMT
//
//


// system include files
#include <memory>
#include <sys/time.h>
#include <cstdlib>

// user include files
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include <TMath.h>
#include <TStyle.h>
#include <TLorentzVector.h>

#include <unordered_map>
//
// class declaration
//

//using namespace edm;
using namespace trigger;
using std::vector;
using std::string;
using std::unordered_map;

class HLTObjectMonitorHI : public DQMEDAnalyzer {
  struct hltPlot
  {
    
    MonitorElement * ME;
    string pathName;
    string pathNameOR;
    string moduleName;
    string moduleNameOR;
    int pathIndex = -99;
    int pathIndexOR = -99;
    string plotLabel;
    string xAxisLabel;
    int nBins;
    double xMin;
    double xMax;
    bool displayInPrimary;
    
  };

   public:
      explicit HLTObjectMonitorHI(const edm::ParameterSet&);
      ~HLTObjectMonitorHI() override;

  //      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      void analyze(const edm::Event&, const edm::EventSetup&) override;
      void bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&) override;
      void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;
      void endRun(edm::Run const&, edm::EventSetup const&) override;
      vector<hltPlot*> plotList;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  double get_wall_time(void);
      // ----------member data ---------------------------

  bool debugPrint;
  HLTConfigProvider hltConfig_;
  string topDirectoryName;
  string mainShifterFolder;
  string backupFolder;
  unordered_map<string, bool> acceptMap;
  unordered_map<hltPlot*, edm::ParameterSet*> plotMap;

  //set Token(-s)
  edm::EDGetTokenT<edm::TriggerResults>   triggerResultsToken_;
  edm::EDGetTokenT<trigger::TriggerEvent> aodTriggerToken_;

  //declare params
  //2018 PbPb run
  edm::ParameterSet HIPUCaloJet40Pt_pset;
  edm::ParameterSet HIPUCaloJet40Eta_pset;
  edm::ParameterSet HIPUCaloJet40Phi_pset;
  edm::ParameterSet HICSPFJet60Pt_pset;
  edm::ParameterSet HICSPFJet60Eta_pset;
  edm::ParameterSet HICSPFJet60Phi_pset;
  edm::ParameterSet HIPUCaloBJet60DeepCSVPt_pset;
  edm::ParameterSet HIPUCaloBJet60DeepCSVEta_pset;
  edm::ParameterSet HIPUCaloBJet60DeepCSVPhi_pset;
  edm::ParameterSet HIPUCaloBJet60CSVv2Pt_pset;
  edm::ParameterSet HIPUCaloBJet60CSVv2Eta_pset;
  edm::ParameterSet HIPUCaloBJet60CSVv2Phi_pset;
  edm::ParameterSet photonPt_pset;
  edm::ParameterSet photonEta_pset;
  edm::ParameterSet photonPhi_pset;
  edm::ParameterSet isphotonPt_pset;
  edm::ParameterSet isphotonEta_pset;
  edm::ParameterSet isphotonPhi_pset;
  edm::ParameterSet electronPt_pset;
  edm::ParameterSet electronEta_pset;
  edm::ParameterSet electronPhi_pset;
  edm::ParameterSet HIL1DoubleMuZMass_pset;
  edm::ParameterSet HIL2DoubleMuZMass_pset;
  edm::ParameterSet HIL3DoubleMuZMass_pset;
  edm::ParameterSet HIL3DoubleMuJpsiMass_pset;
  edm::ParameterSet HIZBSinglePixelPt_pset;
  edm::ParameterSet HIZBSinglePixelEta_pset;
  edm::ParameterSet HIZBSinglePixelPhi_pset;
  edm::ParameterSet HIHFVetoInORSinglePixelMaxTrackPt_pset;
  edm::ParameterSet HIHFVetoInORSinglePixelMaxTrackEta_pset;
  edm::ParameterSet HIHFVetoInORSinglePixelMaxTrackPhi_pset;
  edm::ParameterSet HIHFVetoInORSinglePixelPt_pset;
  edm::ParameterSet HIHFVetoInORSinglePixelEta_pset;
  edm::ParameterSet HIHFVetoInORSinglePixelPhi_pset;
  edm::ParameterSet HISingleEG5HFVetoInAndSinglePixelMaxTrackPt_pset;
  edm::ParameterSet HISingleEG5HFVetoInAndSinglePixelMaxTrackEta_pset;
  edm::ParameterSet HISingleEG5HFVetoInAndSinglePixelMaxTrackPhi_pset;
  edm::ParameterSet HIMuOpenHFVetoInORMaxTrackPt_pset;
  edm::ParameterSet HIMuOpenHFVetoInORMaxTrackEta_pset;
  edm::ParameterSet HIMuOpenHFVetoInORMaxTrackPhi_pset;
  edm::ParameterSet HIMu0HFVetoInAndMaxTrackPt_pset;
  edm::ParameterSet HIMu0HFVetoInAndMaxTrackEta_pset;
  edm::ParameterSet HIMu0HFVetoInAndMaxTrackPhi_pset;
  edm::ParameterSet HIDoubleMuOpenHFVetoInAndMaxTrackPt_pset;
  edm::ParameterSet HIDoubleMuOpenHFVetoInAndMaxTrackEta_pset;
  edm::ParameterSet HIDoubleMuOpenHFVetoInAndMaxTrackPhi_pset;
  edm::ParameterSet HIDoubleMuOpenHFVetoInAndMaxTrackMass_pset;
  edm::ParameterSet HITktkDzeroPt_pset;
  edm::ParameterSet HITktkDzeroEta_pset;
  edm::ParameterSet HITktkDzeroPhi_pset;
  edm::ParameterSet HITktkDzeroMass_pset;
  edm::ParameterSet HITktktkDsPt_pset;
  edm::ParameterSet HITktktkDsEta_pset;
  edm::ParameterSet HITktktkDsPhi_pset;
  edm::ParameterSet HITktktkDsMass_pset;
  edm::ParameterSet HITktktkLcPt_pset;
  edm::ParameterSet HITktktkLcEta_pset;
  edm::ParameterSet HITktktkLcPhi_pset;
  edm::ParameterSet HITktktkLcMass_pset;
  edm::ParameterSet wallTime_pset;

  string processName_;

  hltPlot HIPUCaloJet40Pt_;
  hltPlot HIPUCaloJet40Eta_;
  hltPlot HIPUCaloJet40Phi_;
  hltPlot HICSPFJet60Pt_;
  hltPlot HICSPFJet60Eta_;
  hltPlot HICSPFJet60Phi_;
  hltPlot HIPUCaloBJet60DeepCSVPt_;
  hltPlot HIPUCaloBJet60DeepCSVEta_;
  hltPlot HIPUCaloBJet60DeepCSVPhi_;
  hltPlot HIPUCaloBJet60CSVv2Pt_;
  hltPlot HIPUCaloBJet60CSVv2Eta_;
  hltPlot HIPUCaloBJet60CSVv2Phi_;
  hltPlot photonPt_;
  hltPlot photonEta_;
  hltPlot photonPhi_;
  hltPlot isphotonPt_;
  hltPlot isphotonEta_;
  hltPlot isphotonPhi_;
  hltPlot electronPt_;
  hltPlot electronEta_;
  hltPlot electronPhi_;
  hltPlot HIL1DoubleMuZMass_;
  hltPlot HIL2DoubleMuZMass_;
  hltPlot HIL3DoubleMuZMass_;
  hltPlot HIL3DoubleMuJpsiMass_;
  hltPlot HIZBSinglePixelPt_;
  hltPlot HIZBSinglePixelEta_;
  hltPlot HIZBSinglePixelPhi_;
  hltPlot HIHFVetoInORSinglePixelMaxTrackPt_;
  hltPlot HIHFVetoInORSinglePixelMaxTrackEta_;
  hltPlot HIHFVetoInORSinglePixelMaxTrackPhi_;
  hltPlot HIHFVetoInORSinglePixelPt_;
  hltPlot HIHFVetoInORSinglePixelEta_;
  hltPlot HIHFVetoInORSinglePixelPhi_;
  hltPlot HISingleEG5HFVetoInAndSinglePixelMaxTrackPt_;
  hltPlot HISingleEG5HFVetoInAndSinglePixelMaxTrackEta_;
  hltPlot HISingleEG5HFVetoInAndSinglePixelMaxTrackPhi_;
  hltPlot HIMuOpenHFVetoInORMaxTrackPt_;
  hltPlot HIMuOpenHFVetoInORMaxTrackEta_;
  hltPlot HIMuOpenHFVetoInORMaxTrackPhi_;
  hltPlot HIMu0HFVetoInAndMaxTrackPt_;
  hltPlot HIMu0HFVetoInAndMaxTrackEta_;
  hltPlot HIMu0HFVetoInAndMaxTrackPhi_;
  hltPlot HIDoubleMuOpenHFVetoInAndMaxTrackPt_;
  hltPlot HIDoubleMuOpenHFVetoInAndMaxTrackEta_;
  hltPlot HIDoubleMuOpenHFVetoInAndMaxTrackPhi_;
  hltPlot HIDoubleMuOpenHFVetoInAndMaxTrackMass_;
  hltPlot HITktkDzeroPt_;  
  hltPlot HITktkDzeroEta_; 
  hltPlot HITktkDzeroPhi_; 
  hltPlot HITktkDzeroMass_;
  hltPlot HITktktkDsPt_;   
  hltPlot HITktktkDsEta_;  
  hltPlot HITktktkDsPhi_;  
  hltPlot HITktktkDsMass_; 
  hltPlot HITktktkLcPt_;   
  hltPlot HITktktkLcEta_;  
  hltPlot HITktktkLcPhi_;  
  hltPlot HITktktkLcMass_; 
  hltPlot wallTime_;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HLTObjectMonitorHI::HLTObjectMonitorHI(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
  debugPrint = false;

  topDirectoryName = "HLT/ObjectMonitor";
  mainShifterFolder = topDirectoryName+"/MainShifter";
  backupFolder = topDirectoryName+"/Backup";

  //parse params
  processName_ = iConfig.getParameter<string>("processName");

  HIPUCaloJet40Pt_pset       = iConfig.getParameter<edm::ParameterSet>("HIPUCaloJet40Pt");
  plotMap[&HIPUCaloJet40Pt_] = &HIPUCaloJet40Pt_pset;
  HIPUCaloJet40Eta_pset      = iConfig.getParameter<edm::ParameterSet>("HIPUCaloJet40Eta");
  plotMap[&HIPUCaloJet40Eta_] = &HIPUCaloJet40Eta_pset;
  HIPUCaloJet40Phi_pset      = iConfig.getParameter<edm::ParameterSet>("HIPUCaloJet40Phi");
  plotMap[&HIPUCaloJet40Phi_] = &HIPUCaloJet40Phi_pset;
  HICSPFJet60Pt_pset         = iConfig.getParameter<edm::ParameterSet>("HICSPFJet60Pt");
  plotMap[&HICSPFJet60Pt_] = &HICSPFJet60Pt_pset;
  HICSPFJet60Eta_pset        = iConfig.getParameter<edm::ParameterSet>("HICSPFJet60Eta");
  plotMap[&HICSPFJet60Eta_] = &HICSPFJet60Eta_pset;
  HICSPFJet60Phi_pset        = iConfig.getParameter<edm::ParameterSet>("HICSPFJet60Phi");
  plotMap[&HICSPFJet60Phi_] = &HICSPFJet60Phi_pset;
  HIPUCaloBJet60DeepCSVPt_pset     = iConfig.getParameter<edm::ParameterSet>("HIPUCaloBJet60DeepCSVPt");
  plotMap[&HIPUCaloBJet60DeepCSVPt_] = &HIPUCaloBJet60DeepCSVPt_pset;
  HIPUCaloBJet60DeepCSVEta_pset    = iConfig.getParameter<edm::ParameterSet>("HIPUCaloBJet60DeepCSVEta");
  plotMap[&HIPUCaloBJet60DeepCSVEta_] = &HIPUCaloBJet60DeepCSVEta_pset;
  HIPUCaloBJet60DeepCSVPhi_pset    = iConfig.getParameter<edm::ParameterSet>("HIPUCaloBJet60DeepCSVPhi");
  plotMap[&HIPUCaloBJet60DeepCSVPhi_] = &HIPUCaloBJet60DeepCSVPhi_pset;
  HIPUCaloBJet60CSVv2Pt_pset       = iConfig.getParameter<edm::ParameterSet>("HIPUCaloBJet60CSVv2Pt");
  plotMap[&HIPUCaloBJet60CSVv2Pt_] = &HIPUCaloBJet60CSVv2Pt_pset;
  HIPUCaloBJet60CSVv2Eta_pset      = iConfig.getParameter<edm::ParameterSet>("HIPUCaloBJet60CSVv2Eta");
  plotMap[&HIPUCaloBJet60CSVv2Eta_] = &HIPUCaloBJet60CSVv2Eta_pset;
  HIPUCaloBJet60CSVv2Phi_pset      = iConfig.getParameter<edm::ParameterSet>("HIPUCaloBJet60CSVv2Phi");
  plotMap[&HIPUCaloBJet60CSVv2Phi_] = &HIPUCaloBJet60CSVv2Phi_pset;
  photonPt_pset           = iConfig.getParameter<edm::ParameterSet>("photonPt");
  plotMap[&photonPt_] = &photonPt_pset;
  photonEta_pset          = iConfig.getParameter<edm::ParameterSet>("photonEta");
  plotMap[&photonEta_] = &photonEta_pset;
  photonPhi_pset          = iConfig.getParameter<edm::ParameterSet>("photonPhi");
  plotMap[&photonPhi_] = &photonPhi_pset;
  isphotonPt_pset         = iConfig.getParameter<edm::ParameterSet>("isphotonPt");
  plotMap[&isphotonPt_] = &isphotonPt_pset;
  isphotonEta_pset        = iConfig.getParameter<edm::ParameterSet>("isphotonEta");
  plotMap[&isphotonEta_] = &isphotonEta_pset;
  isphotonPhi_pset        = iConfig.getParameter<edm::ParameterSet>("isphotonPhi");
  plotMap[&isphotonPhi_] = &isphotonPhi_pset;
  electronPt_pset         = iConfig.getParameter<edm::ParameterSet>("electronPt");
  plotMap[&electronPt_] = &electronPt_pset;
  electronEta_pset        = iConfig.getParameter<edm::ParameterSet>("electronEta");
  plotMap[&electronEta_] = &electronEta_pset;
  electronPhi_pset        = iConfig.getParameter<edm::ParameterSet>("electronPhi");
  plotMap[&electronPhi_] = &electronPhi_pset;
  HIL1DoubleMuZMass_pset  = iConfig.getParameter<edm::ParameterSet>("HIL1DoubleMuZMass");
  plotMap[&HIL1DoubleMuZMass_] = &HIL1DoubleMuZMass_pset;
  HIL2DoubleMuZMass_pset  = iConfig.getParameter<edm::ParameterSet>("HIL2DoubleMuZMass");
  plotMap[&HIL2DoubleMuZMass_] = &HIL2DoubleMuZMass_pset;
  HIL3DoubleMuZMass_pset  = iConfig.getParameter<edm::ParameterSet>("HIL3DoubleMuZMass");
  plotMap[&HIL3DoubleMuZMass_] = &HIL3DoubleMuZMass_pset;
  HIL3DoubleMuJpsiMass_pset  = iConfig.getParameter<edm::ParameterSet>("HIL3DoubleMuJpsiMass");
  plotMap[&HIL3DoubleMuJpsiMass_] = &HIL3DoubleMuJpsiMass_pset;
  HIZBSinglePixelPt_pset = iConfig.getParameter<edm::ParameterSet>("HIZBSinglePixelPt");
  plotMap[&HIZBSinglePixelPt_] = &HIZBSinglePixelPt_pset;
  HIZBSinglePixelEta_pset = iConfig.getParameter<edm::ParameterSet>("HIZBSinglePixelEta");
  plotMap[&HIZBSinglePixelEta_] = &HIZBSinglePixelEta_pset;
  HIZBSinglePixelPhi_pset = iConfig.getParameter<edm::ParameterSet>("HIZBSinglePixelPhi");
  plotMap[&HIZBSinglePixelPhi_] = &HIZBSinglePixelPhi_pset;
  HIHFVetoInORSinglePixelMaxTrackPt_pset = iConfig.getParameter<edm::ParameterSet>("HIHFVetoInORSinglePixelMaxTrackPt");
  plotMap[&HIHFVetoInORSinglePixelMaxTrackPt_] = &HIHFVetoInORSinglePixelMaxTrackPt_pset;
  HIHFVetoInORSinglePixelMaxTrackEta_pset = iConfig.getParameter<edm::ParameterSet>("HIHFVetoInORSinglePixelMaxTrackEta");
  plotMap[&HIHFVetoInORSinglePixelMaxTrackEta_] = &HIHFVetoInORSinglePixelMaxTrackEta_pset;
  HIHFVetoInORSinglePixelMaxTrackPhi_pset = iConfig.getParameter<edm::ParameterSet>("HIHFVetoInORSinglePixelMaxTrackPhi");
  plotMap[&HIHFVetoInORSinglePixelMaxTrackPhi_] = &HIHFVetoInORSinglePixelMaxTrackPhi_pset;
  HIHFVetoInORSinglePixelPt_pset = iConfig.getParameter<edm::ParameterSet>("HIHFVetoInORSinglePixelPt");
  plotMap[&HIHFVetoInORSinglePixelPt_] = &HIHFVetoInORSinglePixelPt_pset;
  HIHFVetoInORSinglePixelEta_pset = iConfig.getParameter<edm::ParameterSet>("HIHFVetoInORSinglePixelEta");
  plotMap[&HIHFVetoInORSinglePixelEta_] = &HIHFVetoInORSinglePixelEta_pset;
  HIHFVetoInORSinglePixelPhi_pset = iConfig.getParameter<edm::ParameterSet>("HIHFVetoInORSinglePixelPhi");
  plotMap[&HIHFVetoInORSinglePixelPhi_] = &HIHFVetoInORSinglePixelPhi_pset;
  HISingleEG5HFVetoInAndSinglePixelMaxTrackPt_pset = iConfig.getParameter<edm::ParameterSet>("HISingleEG5HFVetoInAndSinglePixelMaxTrackPt");
  plotMap[&HISingleEG5HFVetoInAndSinglePixelMaxTrackPt_] = &HISingleEG5HFVetoInAndSinglePixelMaxTrackPt_pset;
  HISingleEG5HFVetoInAndSinglePixelMaxTrackEta_pset = iConfig.getParameter<edm::ParameterSet>("HISingleEG5HFVetoInAndSinglePixelMaxTrackEta");
  plotMap[&HISingleEG5HFVetoInAndSinglePixelMaxTrackEta_] = &HISingleEG5HFVetoInAndSinglePixelMaxTrackEta_pset;
  HISingleEG5HFVetoInAndSinglePixelMaxTrackPhi_pset = iConfig.getParameter<edm::ParameterSet>("HISingleEG5HFVetoInAndSinglePixelMaxTrackPhi");
  plotMap[&HISingleEG5HFVetoInAndSinglePixelMaxTrackPhi_] = &HISingleEG5HFVetoInAndSinglePixelMaxTrackPhi_pset;
  HIMuOpenHFVetoInORMaxTrackPt_pset = iConfig.getParameter<edm::ParameterSet>("HIMuOpenHFVetoInORMaxTrackPt");
  plotMap[&HIMuOpenHFVetoInORMaxTrackPt_] = &HIMuOpenHFVetoInORMaxTrackPt_pset;
  HIMuOpenHFVetoInORMaxTrackEta_pset = iConfig.getParameter<edm::ParameterSet>("HIMuOpenHFVetoInORMaxTrackEta");
  plotMap[&HIMuOpenHFVetoInORMaxTrackEta_] = &HIMuOpenHFVetoInORMaxTrackEta_pset;
  HIMuOpenHFVetoInORMaxTrackPhi_pset = iConfig.getParameter<edm::ParameterSet>("HIMuOpenHFVetoInORMaxTrackPhi");
  plotMap[&HIMuOpenHFVetoInORMaxTrackPhi_] = &HIMuOpenHFVetoInORMaxTrackPhi_pset;
  HIMu0HFVetoInAndMaxTrackPt_pset = iConfig.getParameter<edm::ParameterSet>("HIMu0HFVetoInAndMaxTrackPt");
  plotMap[&HIMu0HFVetoInAndMaxTrackPt_] = &HIMu0HFVetoInAndMaxTrackPt_pset;
  HIMu0HFVetoInAndMaxTrackEta_pset = iConfig.getParameter<edm::ParameterSet>("HIMu0HFVetoInAndMaxTrackEta");
  plotMap[&HIMu0HFVetoInAndMaxTrackEta_] = &HIMu0HFVetoInAndMaxTrackEta_pset;
  HIMu0HFVetoInAndMaxTrackPhi_pset = iConfig.getParameter<edm::ParameterSet>("HIMu0HFVetoInAndMaxTrackPhi");
  plotMap[&HIMu0HFVetoInAndMaxTrackPhi_] = &HIMu0HFVetoInAndMaxTrackPhi_pset;
  HIDoubleMuOpenHFVetoInAndMaxTrackPt_pset = iConfig.getParameter<edm::ParameterSet>("HIDoubleMuOpenHFVetoInAndMaxTrackPt");
  plotMap[&HIDoubleMuOpenHFVetoInAndMaxTrackPt_] = &HIDoubleMuOpenHFVetoInAndMaxTrackPt_pset;
  HIDoubleMuOpenHFVetoInAndMaxTrackEta_pset = iConfig.getParameter<edm::ParameterSet>("HIDoubleMuOpenHFVetoInAndMaxTrackEta");
  plotMap[&HIDoubleMuOpenHFVetoInAndMaxTrackEta_] = &HIDoubleMuOpenHFVetoInAndMaxTrackEta_pset;
  HIDoubleMuOpenHFVetoInAndMaxTrackMass_pset = iConfig.getParameter<edm::ParameterSet>("HIDoubleMuOpenHFVetoInAndMaxTrackMass");
  plotMap[&HIDoubleMuOpenHFVetoInAndMaxTrackMass_] = &HIDoubleMuOpenHFVetoInAndMaxTrackMass_pset;
  HITktkDzeroPt_pset = iConfig.getParameter<edm::ParameterSet>("HITktkDzeroPt");
  plotMap[&HITktkDzeroPt_] = &HITktkDzeroPt_pset; 
  HITktkDzeroEta_pset = iConfig.getParameter<edm::ParameterSet>("HITktkDzeroEta");
  plotMap[&HITktkDzeroEta_] = &HITktkDzeroEta_pset; 
  HITktkDzeroPhi_pset = iConfig.getParameter<edm::ParameterSet>("HITktkDzeroPhi");
  plotMap[&HITktkDzeroPhi_] = &HITktkDzeroPhi_pset; 
  HITktkDzeroMass_pset = iConfig.getParameter<edm::ParameterSet>("HITktkDzeroMass");
  plotMap[&HITktkDzeroMass_] = &HITktkDzeroMass_pset;
  HITktktkDsPt_pset = iConfig.getParameter<edm::ParameterSet>("HITktktkDsPt");
  plotMap[&HITktktkDsPt_] = &HITktktkDsPt_pset;  
  HITktktkDsEta_pset = iConfig.getParameter<edm::ParameterSet>("HITktktkDsEta");
  plotMap[&HITktktkDsEta_] = &HITktktkDsEta_pset; 
  HITktktkDsPhi_pset = iConfig.getParameter<edm::ParameterSet>("HITktktkDsPhi");
  plotMap[&HITktktkDsPhi_] = &HITktktkDsPhi_pset; 
  HITktktkDsMass_pset = iConfig.getParameter<edm::ParameterSet>("HITktktkDsMass");
  plotMap[&HITktktkDsMass_] = &HITktktkDsMass_pset; 
  HITktktkLcPt_pset = iConfig.getParameter<edm::ParameterSet>("HITktktkLcPt");
  plotMap[&HITktktkLcPt_] = &HITktktkLcPt_pset;  
  HITktktkLcEta_pset = iConfig.getParameter<edm::ParameterSet>("HITktktkLcEta");
  plotMap[&HITktktkLcEta_] = &HITktktkLcEta_pset; 
  HITktktkLcPhi_pset = iConfig.getParameter<edm::ParameterSet>("HITktktkLcPhi");
  plotMap[&HITktktkLcPhi_] = &HITktktkLcPhi_pset; 
  HITktktkLcMass_pset = iConfig.getParameter<edm::ParameterSet>("HITktktkLcMass");
  plotMap[&HITktktkLcMass_] = &HITktktkLcMass_pset; 

  wallTime_pset = iConfig.getParameter<edm::ParameterSet>("wallTime");
  plotMap[&wallTime_] = &wallTime_pset;
  

  for (auto item = plotMap.begin(); item != plotMap.end(); item++)
    {
      (*item->first).pathName = (*item->second).getParameter<string>("pathName");
      (*item->first).moduleName = (*item->second).getParameter<string>("moduleName");
      (*item->first).nBins = (*item->second).getParameter<int>("NbinsX");
      (*item->first).xMin = (*item->second).getParameter<double>("Xmin");
      (*item->first).xMax = (*item->second).getParameter<double>("Xmax");
      (*item->first).xAxisLabel = (*item->second).getParameter<string>("axisLabel");
      (*item->first).plotLabel =  (*item->second).getParameter<string>("plotLabel");
      (*item->first).displayInPrimary = (*item->second).getParameter<bool>("mainWorkspace");

      if ((*item->second).exists("pathName_OR"))
	{
	  (*item->first).pathNameOR = (*item->second).getParameter<string>("pathName_OR");
	}
      if ((*item->second).exists("moduleName_OR"))
	{
	  (*item->first).moduleNameOR = (*item->second).getParameter<string>("moduleName_OR");
	}

      plotList.push_back(item->first);
    }
  plotMap.clear();
  
  //set Token(s)
  triggerResultsToken_ = consumes<edm::TriggerResults>   (iConfig.getParameter<edm::InputTag>("triggerResults"));
  aodTriggerToken_     = consumes<trigger::TriggerEvent> (iConfig.getParameter<edm::InputTag>("triggerEvent"));

}


HLTObjectMonitorHI::~HLTObjectMonitorHI()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
HLTObjectMonitorHI::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  double start = get_wall_time();

  using namespace edm;

   if (debugPrint) std::cout << "Inside analyze(). " << std::endl;

   // access trigger results
   edm::Handle<edm::TriggerResults> triggerResults;
   iEvent.getByToken(triggerResultsToken_, triggerResults);
   if (!triggerResults.isValid()) return;

   edm::Handle<trigger::TriggerEvent> aodTriggerEvent;
   iEvent.getByToken(aodTriggerToken_, aodTriggerEvent);
   if (!aodTriggerEvent.isValid()) return;

   //reset everything to not accepted at beginning of each event
   unordered_map<string, bool> firedMap = acceptMap;
   for (auto plot: plotList) //loop over paths
     {

       //std::cout<<"CHECK1"<<std::endl;
       if (firedMap[plot->pathName]) continue;
       bool triggerAccept = false;
       const TriggerObjectCollection objects = aodTriggerEvent->getObjects();
       edm::InputTag moduleFilter;
       std::string pathName;
       //std::cout<<"1: "<<triggerResults->accept(plot->pathIndex)<<std::endl;
       //std::cout<<"2: "<<hltConfig_.saveTags(plot->moduleName)<<std::endl;
       if(plot->pathIndex > 0 && triggerResults->accept(plot->pathIndex) && hltConfig_.saveTags(plot->moduleName))
	 {
	   moduleFilter = edm::InputTag(plot->moduleName,"",processName_);
	   pathName = plot->pathName;
	   triggerAccept = true;
	 }
       else if(plot->pathIndexOR > 0 && triggerResults->accept(plot->pathIndexOR) && hltConfig_.saveTags(plot->moduleNameOR))
	 {
	   if (firedMap[plot->pathNameOR]) continue;
	   moduleFilter = edm::InputTag(plot->moduleNameOR,"",processName_);
	   pathName = plot->pathNameOR;
	   triggerAccept = true;
	 }

       if (triggerAccept)
   	 {

           //std::cout<<"CHECK2"<<std::endl;
	   unsigned int moduleFilterIndex = aodTriggerEvent->filterIndex(moduleFilter);
	   
	   if (moduleFilterIndex+1 > aodTriggerEvent->sizeFilters()) return;
	   const Keys &keys = aodTriggerEvent->filterKeys( moduleFilterIndex );

	   ////////////////////////////////
	   ///
	   /// single-object plots
	   ///
	   ////////////////////////////////

	   //calo AK4 jet pt + eta + phi
	   if (pathName == HIPUCaloJet40Pt_.pathName){
	      for (const auto & key : keys){
		 HIPUCaloJet40Pt_.ME->Fill(objects[key].pt());
		 HIPUCaloJet40Eta_.ME->Fill(objects[key].eta());
		 HIPUCaloJet40Phi_.ME->Fill(objects[key].phi());
	      }
	   }
	   //cspf jet pt + eta + phi
	   else if (pathName == HICSPFJet60Pt_.pathName){
	      for (const auto & key : keys){
		 HICSPFJet60Pt_.ME->Fill(objects[key].pt());
		 HICSPFJet60Eta_.ME->Fill(objects[key].eta());
		 HICSPFJet60Phi_.ME->Fill(objects[key].phi());
	      }
	   }
	   //calo Bjet DeepCSV pt + eta + phi
	   else if (pathName == HIPUCaloBJet60DeepCSVPt_.pathName){
	      for (const auto & key : keys){
		 HIPUCaloBJet60DeepCSVPt_.ME->Fill(objects[key].pt());
		 HIPUCaloBJet60DeepCSVEta_.ME->Fill(objects[key].eta());
		 HIPUCaloBJet60DeepCSVPhi_.ME->Fill(objects[key].phi());
	      }
	   }
	   //calo Bjet CSV pt + eta + phi
	   else if (pathName == HIPUCaloBJet60CSVv2Pt_.pathName){
	      for (const auto & key : keys){
		 HIPUCaloBJet60CSVv2Pt_.ME->Fill(objects[key].pt());
		 HIPUCaloBJet60CSVv2Eta_.ME->Fill(objects[key].eta());
		 HIPUCaloBJet60CSVv2Phi_.ME->Fill(objects[key].phi());
	      }
	   }
	   //photon pt + eta + phi
	   else if (pathName == photonPt_.pathName){
	      for (const auto & key : keys){
		 photonPt_.ME->Fill(objects[key].pt());
		 photonEta_.ME->Fill(objects[key].eta());
		 photonPhi_.ME->Fill(objects[key].phi());
	      }
	   }
	   //Islandphoton pt + eta + phi
	   else if (pathName == isphotonPt_.pathName){
	      for (const auto & key : keys){
		 isphotonPt_.ME->Fill(objects[key].pt());
		 isphotonEta_.ME->Fill(objects[key].eta());
		 isphotonPhi_.ME->Fill(objects[key].phi());
	      }
	   }
	   //electron pt + eta + phi
	   else if (pathName == electronPt_.pathName){
	      for (const auto & key : keys){
		 electronPt_.ME->Fill(objects[key].pt());
		 electronEta_.ME->Fill(objects[key].eta());
		 electronPhi_.ME->Fill(objects[key].phi());
	      }
	   }
	   //HIZBSinglePixel pt + eta + phi
	   else if (pathName == HIZBSinglePixelPt_.pathName){
	      for (const auto & key : keys){
		 HIZBSinglePixelPt_.ME->Fill(objects[key].pt());
		 HIZBSinglePixelEta_.ME->Fill(objects[key].eta());
		 HIZBSinglePixelPhi_.ME->Fill(objects[key].phi());
	      }
	   }
	   //HIHFVetoInORSinglePixelMaxTrack pt + eta + phi
	   else if (pathName == HIHFVetoInORSinglePixelMaxTrackPt_.pathName){
	      for (const auto & key : keys){
		 HIHFVetoInORSinglePixelMaxTrackPt_.ME->Fill(objects[key].pt());
		 HIHFVetoInORSinglePixelMaxTrackEta_.ME->Fill(objects[key].eta());
		 HIHFVetoInORSinglePixelMaxTrackPhi_.ME->Fill(objects[key].phi());
	      }
	   }
	   //HIHFVetoInORSinglePixel pt + eta + phi
	   else if (pathName == HIHFVetoInORSinglePixelPt_.pathName){
	      for (const auto & key : keys){
		 HIHFVetoInORSinglePixelPt_.ME->Fill(objects[key].pt());
		 HIHFVetoInORSinglePixelEta_.ME->Fill(objects[key].eta());
		 HIHFVetoInORSinglePixelPhi_.ME->Fill(objects[key].phi());
	      }
	   }
	   //HISingleEG5HFVetoInAndSinglePixelMaxTrack pt + eta + phi
	   else if (pathName == HISingleEG5HFVetoInAndSinglePixelMaxTrackPt_.pathName){
	      for (const auto & key : keys){
		 HISingleEG5HFVetoInAndSinglePixelMaxTrackPt_.ME->Fill(objects[key].pt());
		 HISingleEG5HFVetoInAndSinglePixelMaxTrackEta_.ME->Fill(objects[key].eta());
		 HISingleEG5HFVetoInAndSinglePixelMaxTrackPhi_.ME->Fill(objects[key].phi());
	      }
	   }
	   //HIMuOpenHFVetoInORMaxTrack pt + eta + phi
	   else if (pathName == HIMuOpenHFVetoInORMaxTrackPt_.pathName){
	      for (const auto & key : keys){
		 HIMuOpenHFVetoInORMaxTrackPt_.ME->Fill(objects[key].pt());
		 HIMuOpenHFVetoInORMaxTrackEta_.ME->Fill(objects[key].eta());
		 HIMuOpenHFVetoInORMaxTrackPhi_.ME->Fill(objects[key].phi());
	      }
	   }
	   //HIMu0HFVetoInAndMaxTrack pt + eta + phi
	   else if (pathName == HIMu0HFVetoInAndMaxTrackPt_.pathName){
	      for (const auto & key : keys){
		 HIMu0HFVetoInAndMaxTrackPt_.ME->Fill(objects[key].pt());
		 HIMu0HFVetoInAndMaxTrackEta_.ME->Fill(objects[key].eta());
		 HIMu0HFVetoInAndMaxTrackPhi_.ME->Fill(objects[key].phi());
	      }
	   }


	   // ////////////////////////////////
	   // ///
	   // /// double-object plots
	   // ///
	   // ////////////////////////////////

	   else if (pathName == HIL1DoubleMuZMass_.pathName){
	      const double mu_mass(.105658);
	      unsigned int kCnt0 = 0;
	      for (const auto & key0: keys){
		 unsigned int kCnt1 = 0;
		 for (const auto & key1: keys){
		    if (key0 != key1 && kCnt1 > kCnt0){ // avoid filling hists with same objs && avoid double counting separate objs
		       // if (abs(objects[key0].id()) == 13 && (objects[key0].id()+objects[key1].id()==0))  // id is not filled for l1 stage2 muons
		       TLorentzVector mu1, mu2, dimu;
		       mu1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), mu_mass);
		       mu2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), mu_mass);
		       dimu = mu1+mu2;
		       if(dimu.M()>HIL1DoubleMuZMass_.xMin && dimu.M()<HIL1DoubleMuZMass_.xMax) HIL1DoubleMuZMass_.ME->Fill(dimu.M());
		    }
		    kCnt1 +=1;
		 }
		 kCnt0 +=1;
	      }
	   }
	   else if (pathName == HIL2DoubleMuZMass_.pathName){
	      const double mu_mass(.105658);
	      unsigned int kCnt0 = 0;
	      for (const auto & key0: keys){
		 unsigned int kCnt1 = 0;
		 for (const auto & key1: keys){
		    if (key0 != key1 && kCnt1 > kCnt0){ // avoid filling hists with same objs && avoid double counting separate objs
		       if (abs(objects[key0].id()) == 13 && (objects[key0].id()+objects[key1].id()==0)){  // check muon id and dimuon charge
			  TLorentzVector mu1, mu2, dimu;
			  mu1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), mu_mass);
			  mu2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), mu_mass);
			  dimu = mu1+mu2;
			  if(dimu.M()>HIL2DoubleMuZMass_.xMin && dimu.M()<HIL2DoubleMuZMass_.xMax) HIL2DoubleMuZMass_.ME->Fill(dimu.M());
		       }
		    }
		    kCnt1 +=1;
		 }
		 kCnt0 +=1;
	      }
	   }
	   else if (pathName == HIL3DoubleMuZMass_.pathName){
	      const double mu_mass(.105658);
	      unsigned int kCnt0 = 0;
	      for (const auto & key0: keys){
		 unsigned int kCnt1 = 0;
		 for (const auto & key1: keys){
		    if (key0 != key1 && kCnt1 > kCnt0){ // avoid filling hists with same objs && avoid double counting separate objs
		       if (abs(objects[key0].id()) == 13 && (objects[key0].id()+objects[key1].id()==0)){  // check muon id and dimuon charge
			  TLorentzVector mu1, mu2, dimu;
			  mu1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), mu_mass);
			  mu2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), mu_mass);
			  dimu = mu1+mu2;
			  if(dimu.M()>HIL3DoubleMuZMass_.xMin && dimu.M()<HIL3DoubleMuZMass_.xMax) HIL3DoubleMuZMass_.ME->Fill(dimu.M());
		       }
		    }
		    kCnt1 +=1;
		 }
		 kCnt0 +=1;
	      }
	   }
	   else if (pathName == HIDoubleMuOpenHFVetoInAndMaxTrackPt_.pathName){
	      const double mu_mass(.105658);
	      unsigned int kCnt0 = 0;
	      for (const auto & key0: keys){
		 unsigned int kCnt1 = 0;
		 for (const auto & key1: keys){
		    if (key0 != key1 && kCnt1 > kCnt0){ // avoid filling hists with same objs && avoid double counting separate objs
		       if (abs(objects[key0].id()) == 13 && (objects[key0].id()+objects[key1].id()==0)){  // check muon id and dimuon charge
			  TLorentzVector mu1, mu2, dimu;
			  mu1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), mu_mass);
			  mu2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), mu_mass);
			  dimu = mu1+mu2;
			  if(dimu.Pt()>HIDoubleMuOpenHFVetoInAndMaxTrackPt_.xMin && dimu.Pt()<HIDoubleMuOpenHFVetoInAndMaxTrackPt_.xMax) HIDoubleMuOpenHFVetoInAndMaxTrackPt_.ME->Fill(dimu.Pt());
			  if(dimu.Eta()>HIDoubleMuOpenHFVetoInAndMaxTrackEta_.xMin && dimu.Eta()<HIDoubleMuOpenHFVetoInAndMaxTrackEta_.xMax) HIDoubleMuOpenHFVetoInAndMaxTrackEta_.ME->Fill(dimu.Eta());
			  if(dimu.M()>HIDoubleMuOpenHFVetoInAndMaxTrackMass_.xMin && dimu.M()<HIDoubleMuOpenHFVetoInAndMaxTrackMass_.xMax) HIDoubleMuOpenHFVetoInAndMaxTrackMass_.ME->Fill(dimu.M());
		       }
		    }
		    kCnt1 +=1;
		 }
		 kCnt0 +=1;
	      }
           }
//HITktkDzeroPt_;
//HITktkDzeroEta_
//HITktkDzeroPhi_
//HITktkDzeroMass
//HITktktkDsPt_; 
//HITktktkDsEta_;
//HITktktkDsPhi_;
//HITktktkDsMass_
//HITktktkLcPt_; 
//HITktktkLcEta_;
//HITktktkLcPhi_;
//HITktktkLcMass_
           else if (pathName == HITktkDzeroPt_.pathName){
             const double pi_mass(0.13957018);
             unsigned int kCnt0 = 0;
             for (const auto & key0: keys){
               unsigned int kCnt1 = 0;
               for (const auto & key1: keys){
                 unsigned int kCnt2 = 0;
                 for (const auto & key2: keys){
                   if (key0 != key1 && kCnt1 > kCnt0 && key0 != key2 && key1 != key2 && kCnt2 > kCnt1){ // avoid filling hists with same objs && avoid double counting separate objs
                     // if (abs(objects[key0].id()) == 13 && (objects[key0].id()+objects[key1].id()==0))  // id is not filled for l1 stage2 muons
                     TLorentzVector mu1, mu2, mu3, dimu;
                     mu1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), pi_mass);
                     mu2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), pi_mass);
                     mu3.SetPtEtaPhiM(objects[key2].pt(), objects[key2].eta(), objects[key2].phi(), pi_mass);
                     dimu = mu1+mu2;
                     if(dimu.Pt()>HITktkDzeroPt_.xMin && dimu.Pt()<HITktkDzeroPt_.xMax) HITktkDzeroPt_.ME->Fill(dimu.Pt());
                     if(dimu.Eta()>HITktkDzeroEta_.xMin && dimu.Eta()<HITktkDzeroEta_.xMax) HITktkDzeroEta_.ME->Fill(dimu.Eta());
                     if(dimu.Phi()>HITktkDzeroPhi_.xMin && dimu.Phi()<HITktkDzeroPhi_.xMax) HITktkDzeroPhi_.ME->Fill(dimu.Eta());
                     if(dimu.M()>HITktkDzeroMass_.xMin && dimu.M()<HITktkDzeroMass_.xMax) HITktkDzeroMass_.ME->Fill(dimu.M());
                   }
                   kCnt2 +=1;
                 }
                 kCnt1 +=1;
               }
               kCnt0 +=1;
             }
           }
           else if (pathName == HITktktkDsPt_.pathName){
             const double pi_mass(0.13957018);
             unsigned int kCnt0 = 0;
             for (const auto & key0: keys){
               unsigned int kCnt1 = 0;
               for (const auto & key1: keys){
                 unsigned int kCnt2 = 0;
                 for (const auto & key2: keys){
                   if (key0 != key1 && kCnt1 > kCnt0 && key0 != key2 && key1 != key2 && kCnt2 > kCnt1){ // avoid filling hists with same objs && avoid double counting separate objs
                     // if (abs(objects[key0].id()) == 13 && (objects[key0].id()+objects[key1].id()==0))  // id is not filled for l1 stage2 muons
                     TLorentzVector mu1, mu2, mu3, dimu;
                     mu1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), pi_mass);
                     mu2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), pi_mass);
                     mu3.SetPtEtaPhiM(objects[key2].pt(), objects[key2].eta(), objects[key2].phi(), pi_mass);
                     dimu = mu1+mu2;
                     if(dimu.Pt()>HITktktkDsPt_.xMin && dimu.Pt()<HITktktkDsPt_.xMax) HITktktkDsPt_.ME->Fill(dimu.Pt());
                     if(dimu.Eta()>HITktktkDsEta_.xMin && dimu.Eta()<HITktktkDsEta_.xMax) HITktktkDsEta_.ME->Fill(dimu.Eta());
                     if(dimu.Phi()>HITktktkDsPhi_.xMin && dimu.Phi()<HITktktkDsPhi_.xMax) HITktktkDsPhi_.ME->Fill(dimu.Eta());
                     if(dimu.M()>HITktktkDsMass_.xMin && dimu.M()<HITktktkDsMass_.xMax) HITktktkDsMass_.ME->Fill(dimu.M());
                   }
                   kCnt2 +=1;
                 }
                 kCnt1 +=1;
               }
               kCnt0 +=1;
             }
           }
           else if (pathName == HITktktkLcPt_.pathName){
             const double pi_mass(0.13957018);
             unsigned int kCnt0 = 0;
             for (const auto & key0: keys){
               unsigned int kCnt1 = 0;
               for (const auto & key1: keys){
                 unsigned int kCnt2 = 0;
                 for (const auto & key2: keys){
                   if (key0 != key1 && kCnt1 > kCnt0 && key0 != key2 && key1 != key2 && kCnt2 > kCnt1){ // avoid filling hists with same objs && avoid double counting separate objs
                     // if (abs(objects[key0].id()) == 13 && (objects[key0].id()+objects[key1].id()==0))  // id is not filled for l1 stage2 muons
                     TLorentzVector mu1, mu2, mu3, dimu;
                     mu1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), pi_mass);
                     mu2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), pi_mass);
                     mu3.SetPtEtaPhiM(objects[key2].pt(), objects[key2].eta(), objects[key2].phi(), pi_mass);
                     dimu = mu1+mu2;
                     if(dimu.Pt()>HITktktkLcPt_.xMin && dimu.Pt()<HITktktkLcPt_.xMax) HITktktkLcPt_.ME->Fill(dimu.Pt());
                     if(dimu.Eta()>HITktktkLcEta_.xMin && dimu.Eta()<HITktktkLcEta_.xMax) HITktktkLcEta_.ME->Fill(dimu.Eta());
                     if(dimu.Phi()>HITktktkLcPhi_.xMin && dimu.Phi()<HITktktkLcPhi_.xMax) HITktktkLcPhi_.ME->Fill(dimu.Eta());
                     if(dimu.M()>HITktktkLcMass_.xMin && dimu.M()<HITktktkLcMass_.xMax) HITktktkLcMass_.ME->Fill(dimu.M());
                   }
                   kCnt2 +=1;
                 }
                 kCnt1 +=1;
               }
               kCnt0 +=1;
             }
           }

	   firedMap[pathName] = true;
	 } //end if trigger accept
     } //end loop over plots/paths

   //   sleep(1); //sleep for 1s, used to calibrate timing
   double end = get_wall_time();
   double wallTime = end - start;
   wallTime_.ME->Fill(wallTime);
}

// ------------ method called when starting to processes a run  ------------
void
HLTObjectMonitorHI::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  if (debugPrint) std::cout << "Calling beginRun. " << std::endl;
  bool changed = true;
  if (hltConfig_.init(iRun, iSetup, processName_, changed))
    {
      if (debugPrint) std::cout << "Extracting HLTconfig. " << std::endl;
    }

  //get path indicies from menu 
  string pathName_noVersion;
  vector<string> triggerPaths = hltConfig_.triggerNames();

  for (const auto & pathName : triggerPaths)
    {
      pathName_noVersion = hltConfig_.removeVersion(pathName);
      for (auto plot : plotList)
  	{	
  	  if (plot->pathName == pathName_noVersion)
  	    {
  	      (*plot).pathIndex = hltConfig_.triggerIndex(pathName);
  	    }
  	  else if (plot->pathNameOR == pathName_noVersion)
  	    {
	      (*plot).pathIndexOR = hltConfig_.triggerIndex(pathName);
  	    }
  	}
    }
  vector<hltPlot*> plotList_temp;
  for (auto plot : plotList)
    {
      if (plot->pathIndex > 0 || plot->pathIndexOR > 0)
	{
	  plotList_temp.push_back(plot);
	  acceptMap[plot->pathName] = false;
	  if (plot->pathIndexOR > 0) acceptMap[plot->pathNameOR] = false;
	}
    }
  //now re-assign plotList to contain only the plots with paths in the menu.
  plotList = plotList_temp;
  plotList_temp.clear();

}

// ------------ method called when ending the processing of a run  ------------

void
HLTObjectMonitorHI::endRun(edm::Run const&, edm::EventSetup const&)
{
  if (debugPrint) std::cout << "Calling endRun. " << std::endl;
}

void HLTObjectMonitorHI::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup)
{

  ////////////////////////////////
  ///
  /// Main shifter workspace plots
  ///
  ////////////////////////////////
  
  //book wall time separately
  ibooker.setCurrentFolder(mainShifterFolder);
  wallTime_.ME = ibooker.book1D(wallTime_.plotLabel,wallTime_.pathName,wallTime_.nBins,wallTime_.xMin,wallTime_.xMax);
  wallTime_.ME->setAxisTitle(wallTime_.xAxisLabel);

  for (auto plot : plotList)
    {
      std::string display_pathNames = plot->pathName;
      if (!plot->pathNameOR.empty()) display_pathNames = plot->pathName + " OR " + plot->pathNameOR;

      if (plot->displayInPrimary)
	{
	  ibooker.setCurrentFolder(mainShifterFolder);
	  (*plot).ME = ibooker.book1D(plot->plotLabel,display_pathNames.c_str(),plot->nBins,plot->xMin,plot->xMax);
	  (*plot).ME->setAxisTitle(plot->xAxisLabel);
	  //need to add OR statement
	}
      else
	{
	  ibooker.setCurrentFolder(backupFolder);
	  (*plot).ME = ibooker.book1D(plot->plotLabel,display_pathNames.c_str(),plot->nBins,plot->xMin,plot->xMax);
	  (*plot).ME->setAxisTitle(plot->xAxisLabel);
	}
    }

}

double HLTObjectMonitorHI::get_wall_time()
{
  struct timeval time;
  if (gettimeofday(&time,nullptr)) return 0;
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
HLTObjectMonitorHI::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
HLTObjectMonitorHI::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
// void
// HLTObjectMonitorHI::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
//   //The following says we do not know what parameters are allowed so do no validation
//   // Please change this to state exactly what you do use, even if it is no parameters
//   edm::ParameterSetDescription desc;
//   desc.setUnknown();
//   descriptions.addDefault(desc);
// }

//define this as a plug-in
DEFINE_FWK_MODULE(HLTObjectMonitorHI);
