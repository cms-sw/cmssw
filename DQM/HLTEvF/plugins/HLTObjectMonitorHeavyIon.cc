// -*- C++ -*-
//
// Package:    DQM/HLTObjectMonitorHeavyIon
// Class:      HLTObjectMonitorHeavyIon
//
/**\class HLTObjectMonitorHeavyIon HLTObjectMonitorHeavyIon.cc DQM/HLTEvF/plugins/HLTObjectMonitorHeavyIon.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Charles Nicholas Mueller
//         Created:  Mon, 9 Nov 2015 20:47:01 GMT
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
#include "DataFormats/Scalers/interface/LumiScalers.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//for collections
#include "HLTrigger/JetMET/interface/AlphaT.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "TMath.h"
#include "TStyle.h"
#include "TLorentzVector.h"

#include <unordered_map>
//
// class declaration
//

//using namespace edm;
using namespace trigger;
using std::vector;
using std::string;
using std::unordered_map;

class HLTObjectMonitorHeavyIon : public DQMEDAnalyzer {
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
      explicit HLTObjectMonitorHeavyIon(const edm::ParameterSet&);
      ~HLTObjectMonitorHeavyIon();

  //      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&) override;
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
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  edm::EDGetTokenT<trigger::TriggerEvent> aodTriggerToken_;
  edm::EDGetTokenT<LumiScalersCollection> lumiScalersToken_;

  //declare params
  edm::ParameterSet wallTime_pset;
  edm::ParameterSet caloJetPt_HI_pset;
  edm::ParameterSet caloJetEta_HI_pset;
  edm::ParameterSet caloJetPhi_HI_pset;
  edm::ParameterSet singlePhotonEta1p5Pt_HI_pset;
  edm::ParameterSet singlePhotonEta1p5Eta_HI_pset;
  edm::ParameterSet singlePhotonEta1p5Phi_HI_pset;
  edm::ParameterSet singlePhotonEta3p1Pt_HI_pset;
  edm::ParameterSet singlePhotonEta3p1Eta_HI_pset;
  edm::ParameterSet singlePhotonEta3p1Phi_HI_pset;
  edm::ParameterSet doublePhotonMass_HI_pset;  
  edm::ParameterSet fullTrack12MinBiasHFPt_HI_pset;
  edm::ParameterSet fullTrack12MinBiasHFEta_HI_pset;
  edm::ParameterSet fullTrack12MinBiasHFPhi_HI_pset;
  edm::ParameterSet fullTrack12Centrality010Pt_HI_pset;
  edm::ParameterSet fullTrack12Centrality010Eta_HI_pset;
  edm::ParameterSet fullTrack12Centrality010Phi_HI_pset;
  edm::ParameterSet fullTrack12Centrality301Pt_HI_pset;
  edm::ParameterSet fullTrack12Centrality301Eta_HI_pset;
  edm::ParameterSet fullTrack12Centrality301Phi_HI_pset;
  edm::ParameterSet dMesonTrackPt_HI_pset;
  edm::ParameterSet dMesonTrackEta_HI_pset;
  edm::ParameterSet dMesonTrackSystemMass_HI_pset;
  edm::ParameterSet dMesonTrackSystemPt_HI_pset;
  edm::ParameterSet l1DoubleMuPt_HI_pset;
  edm::ParameterSet l1DoubleMuEta_HI_pset;
  edm::ParameterSet l1DoubleMuPhi_HI_pset;
  edm::ParameterSet l3DoubleMuPt_HI_pset;
  edm::ParameterSet l3DoubleMuEta_HI_pset;
  edm::ParameterSet l3DoubleMuPhi_HI_pset;
  edm::ParameterSet l2SingleMuPt_HI_pset;
  edm::ParameterSet l2SingleMuEta_HI_pset;
  edm::ParameterSet l2SingleMuPhi_HI_pset;
  edm::ParameterSet l3SingleMuPt_HI_pset;
  edm::ParameterSet l3SingleMuEta_HI_pset;
  edm::ParameterSet l3SingleMuPhi_HI_pset;
  edm::ParameterSet l3SingleMuHFPt_HI_pset;
  edm::ParameterSet l3SingleMuHFEta_HI_pset;
  edm::ParameterSet l3SingleMuHFPhi_HI_pset;
  edm::ParameterSet fullTrack24Pt_HI_pset;
  edm::ParameterSet fullTrack24Eta_HI_pset;
  edm::ParameterSet fullTrack24Phi_HI_pset;
  edm::ParameterSet fullTrack24Centrality301Pt_HI_pset;
  edm::ParameterSet fullTrack24Centrality301Eta_HI_pset;
  edm::ParameterSet fullTrack24Centrality301Phi_HI_pset;
  edm::ParameterSet fullTrack34Pt_HI_pset;
  edm::ParameterSet fullTrack34Eta_HI_pset;
  edm::ParameterSet fullTrack34Phi_HI_pset;
  edm::ParameterSet fullTrack34Centrality301Pt_HI_pset;
  edm::ParameterSet fullTrack34Centrality301Eta_HI_pset;
  edm::ParameterSet fullTrack34Centrality301Phi_HI_pset;
  //pp ref
  edm::ParameterSet caloJetRefPt_HI_pset;
  edm::ParameterSet caloJetRefEta_HI_pset;
  edm::ParameterSet caloJetRefPhi_HI_pset;
  edm::ParameterSet dMesonRefTrackPt_HI_pset;
  edm::ParameterSet dMesonRefTrackEta_HI_pset;
  edm::ParameterSet dMesonRefTrackSystemMass_HI_pset;
  edm::ParameterSet dMesonRefTrackSystemPt_HI_pset;


  string processName_;

  hltPlot wallTime_;
  hltPlot caloJetPt_HI_;
  hltPlot caloJetEta_HI_;
  hltPlot caloJetPhi_HI_;
  hltPlot singlePhotonEta1p5Pt_HI_;
  hltPlot singlePhotonEta1p5Eta_HI_;
  hltPlot singlePhotonEta1p5Phi_HI_;
  hltPlot singlePhotonEta3p1Pt_HI_;
  hltPlot singlePhotonEta3p1Eta_HI_;
  hltPlot singlePhotonEta3p1Phi_HI_;
  hltPlot doublePhotonMass_HI_; 
  hltPlot fullTrack12MinBiasHFPt_HI_;
  hltPlot fullTrack12MinBiasHFEta_HI_;
  hltPlot fullTrack12MinBiasHFPhi_HI_;
  hltPlot fullTrack12Centrality010Pt_HI_;
  hltPlot fullTrack12Centrality010Eta_HI_;
  hltPlot fullTrack12Centrality010Phi_HI_;
  hltPlot fullTrack12Centrality301Pt_HI_;
  hltPlot fullTrack12Centrality301Eta_HI_;
  hltPlot fullTrack12Centrality301Phi_HI_;
  hltPlot dMesonTrackPt_HI_;
  hltPlot dMesonTrackEta_HI_;
  hltPlot dMesonTrackSystemMass_HI_;
  hltPlot dMesonTrackSystemPt_HI_;
  hltPlot l1DoubleMuPt_HI_;
  hltPlot l1DoubleMuEta_HI_;
  hltPlot l1DoubleMuPhi_HI_;
  hltPlot l3DoubleMuPt_HI_;
  hltPlot l3DoubleMuEta_HI_;
  hltPlot l3DoubleMuPhi_HI_;
  hltPlot l2SingleMuPt_HI_;
  hltPlot l2SingleMuEta_HI_;
  hltPlot l2SingleMuPhi_HI_;
  hltPlot l3SingleMuPt_HI_;
  hltPlot l3SingleMuEta_HI_;
  hltPlot l3SingleMuPhi_HI_;
  hltPlot l3SingleMuHFPt_HI_;
  hltPlot l3SingleMuHFEta_HI_;
  hltPlot l3SingleMuHFPhi_HI_;
  hltPlot fullTrack24Pt_HI_;
  hltPlot fullTrack24Eta_HI_;
  hltPlot fullTrack24Phi_HI_;
  hltPlot fullTrack24Centrality301Pt_HI_;
  hltPlot fullTrack24Centrality301Eta_HI_;
  hltPlot fullTrack24Centrality301Phi_HI_;
  hltPlot fullTrack34Pt_HI_;
  hltPlot fullTrack34Eta_HI_;
  hltPlot fullTrack34Phi_HI_;
  hltPlot fullTrack34Centrality301Pt_HI_;
  hltPlot fullTrack34Centrality301Eta_HI_;
  hltPlot fullTrack34Centrality301Phi_HI_;
  //pp ref
  hltPlot caloJetRefPt_HI_;
  hltPlot caloJetRefEta_HI_;
  hltPlot caloJetRefPhi_HI_;
  hltPlot dMesonRefTrackPt_HI_;
  hltPlot dMesonRefTrackEta_HI_;
  hltPlot dMesonRefTrackSystemMass_HI_;
  hltPlot dMesonRefTrackSystemPt_HI_;
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
HLTObjectMonitorHeavyIon::HLTObjectMonitorHeavyIon(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
  debugPrint = false;

  topDirectoryName = "HLT/ObjectMonitor";
  mainShifterFolder = topDirectoryName+"/MainShifter";
  backupFolder = topDirectoryName+"/Backup";

  //parse params
  processName_ = iConfig.getParameter<string>("processName");

  wallTime_pset = iConfig.getParameter<edm::ParameterSet>("wallTime");
  plotMap[&wallTime_] = &wallTime_pset;
  caloJetPt_HI_pset = iConfig.getParameter<edm::ParameterSet>("caloJetPt_HI");
  plotMap[&caloJetPt_HI_] = &caloJetPt_HI_pset;
  caloJetEta_HI_pset = iConfig.getParameter<edm::ParameterSet>("caloJetEta_HI");
  plotMap[&caloJetEta_HI_] = &caloJetEta_HI_pset;
  caloJetPhi_HI_pset = iConfig.getParameter<edm::ParameterSet>("caloJetPhi_HI");
  plotMap[&caloJetPhi_HI_] = &caloJetPhi_HI_pset;
  singlePhotonEta1p5Pt_HI_pset = iConfig.getParameter<edm::ParameterSet>("singlePhotonEta1p5Pt_HI");
  plotMap[&singlePhotonEta1p5Pt_HI_] = &singlePhotonEta1p5Pt_HI_pset;
  singlePhotonEta1p5Eta_HI_pset = iConfig.getParameter<edm::ParameterSet>("singlePhotonEta1p5Eta_HI");
  plotMap[&singlePhotonEta1p5Eta_HI_] = &singlePhotonEta1p5Eta_HI_pset;
  singlePhotonEta1p5Phi_HI_pset = iConfig.getParameter<edm::ParameterSet>("singlePhotonEta1p5Phi_HI");
  plotMap[&singlePhotonEta1p5Phi_HI_] = &singlePhotonEta1p5Phi_HI_pset;
  singlePhotonEta3p1Pt_HI_pset = iConfig.getParameter<edm::ParameterSet>("singlePhotonEta3p1Pt_HI");
  plotMap[&singlePhotonEta3p1Pt_HI_] = &singlePhotonEta3p1Pt_HI_pset;
  singlePhotonEta3p1Eta_HI_pset = iConfig.getParameter<edm::ParameterSet>("singlePhotonEta3p1Eta_HI");
  plotMap[&singlePhotonEta3p1Eta_HI_] = &singlePhotonEta3p1Eta_HI_pset;
  singlePhotonEta3p1Phi_HI_pset = iConfig.getParameter<edm::ParameterSet>("singlePhotonEta3p1Phi_HI");
  plotMap[&singlePhotonEta3p1Phi_HI_] = &singlePhotonEta3p1Phi_HI_pset;
  doublePhotonMass_HI_pset = iConfig.getParameter<edm::ParameterSet>("doublePhotonMass_HI");
  plotMap[&doublePhotonMass_HI_] = &doublePhotonMass_HI_pset;
  fullTrack12MinBiasHFPt_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack12MinBiasHFPt_HI");
  plotMap[&fullTrack12MinBiasHFPt_HI_] = &fullTrack12MinBiasHFPt_HI_pset;
  fullTrack12MinBiasHFEta_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack12MinBiasHFEta_HI");
  plotMap[&fullTrack12MinBiasHFEta_HI_] = &fullTrack12MinBiasHFEta_HI_pset;
  fullTrack12MinBiasHFPhi_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack12MinBiasHFPhi_HI");
  plotMap[&fullTrack12MinBiasHFPhi_HI_] = &fullTrack12MinBiasHFPhi_HI_pset;
  fullTrack12Centrality010Pt_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack12Centrality010Pt_HI");
  plotMap[&fullTrack12Centrality010Pt_HI_] = &fullTrack12Centrality010Pt_HI_pset;
  fullTrack12Centrality010Eta_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack12Centrality010Eta_HI");
  plotMap[&fullTrack12Centrality010Eta_HI_] = &fullTrack12Centrality010Eta_HI_pset;
  fullTrack12Centrality010Phi_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack12Centrality010Phi_HI");
  plotMap[&fullTrack12Centrality010Phi_HI_] = &fullTrack12Centrality010Phi_HI_pset;
  fullTrack12Centrality301Pt_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack12Centrality301Pt_HI");
  plotMap[&fullTrack12Centrality301Pt_HI_] = &fullTrack12Centrality301Pt_HI_pset;
  fullTrack12Centrality301Eta_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack12Centrality301Eta_HI");
  plotMap[&fullTrack12Centrality301Eta_HI_] = &fullTrack12Centrality301Eta_HI_pset;
  fullTrack12Centrality301Phi_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack12Centrality301Phi_HI");
  plotMap[&fullTrack12Centrality301Phi_HI_] = &fullTrack12Centrality301Phi_HI_pset;
  dMesonTrackPt_HI_pset = iConfig.getParameter<edm::ParameterSet>("dMesonTrackPt_HI");
  plotMap[&dMesonTrackPt_HI_] = &dMesonTrackPt_HI_pset;
  dMesonTrackEta_HI_pset = iConfig.getParameter<edm::ParameterSet>("dMesonTrackEta_HI");
  plotMap[&dMesonTrackEta_HI_] = &dMesonTrackEta_HI_pset;
  dMesonTrackSystemMass_HI_pset = iConfig.getParameter<edm::ParameterSet>("dMesonTrackSystemMass_HI");
  plotMap[&dMesonTrackSystemMass_HI_] = &dMesonTrackSystemMass_HI_pset;
  dMesonTrackSystemPt_HI_pset = iConfig.getParameter<edm::ParameterSet>("dMesonTrackSystemPt_HI");
  plotMap[&dMesonTrackSystemPt_HI_] = &dMesonTrackSystemPt_HI_pset;
  l1DoubleMuPt_HI_pset = iConfig.getParameter<edm::ParameterSet>("l1DoubleMuPt_HI");
  plotMap[&l1DoubleMuPt_HI_] = &l1DoubleMuPt_HI_pset;
  l1DoubleMuEta_HI_pset = iConfig.getParameter<edm::ParameterSet>("l1DoubleMuEta_HI");
  plotMap[&l1DoubleMuEta_HI_] = &l1DoubleMuEta_HI_pset;
  l1DoubleMuPhi_HI_pset = iConfig.getParameter<edm::ParameterSet>("l1DoubleMuPhi_HI");
  plotMap[&l1DoubleMuPhi_HI_] = &l1DoubleMuPhi_HI_pset;
  l3DoubleMuPt_HI_pset = iConfig.getParameter<edm::ParameterSet>("l3DoubleMuPt_HI");
  plotMap[&l3DoubleMuPt_HI_] = &l3DoubleMuPt_HI_pset;
  l3DoubleMuEta_HI_pset = iConfig.getParameter<edm::ParameterSet>("l3DoubleMuEta_HI");
  plotMap[&l3DoubleMuEta_HI_] = &l3DoubleMuEta_HI_pset;
  l3DoubleMuPhi_HI_pset = iConfig.getParameter<edm::ParameterSet>("l3DoubleMuPhi_HI");
  plotMap[&l3DoubleMuPhi_HI_] = &l3DoubleMuPhi_HI_pset;
  l2SingleMuPt_HI_pset = iConfig.getParameter<edm::ParameterSet>("l2SingleMuPt_HI");
  plotMap[&l2SingleMuPt_HI_] = &l2SingleMuPt_HI_pset;
  l2SingleMuEta_HI_pset = iConfig.getParameter<edm::ParameterSet>("l2SingleMuEta_HI");
  plotMap[&l2SingleMuEta_HI_] = &l2SingleMuEta_HI_pset;
  l2SingleMuPhi_HI_pset = iConfig.getParameter<edm::ParameterSet>("l2SingleMuPhi_HI");
  plotMap[&l2SingleMuPhi_HI_] = &l2SingleMuPhi_HI_pset;
  l3SingleMuPt_HI_pset = iConfig.getParameter<edm::ParameterSet>("l3SingleMuPt_HI");
  plotMap[&l3SingleMuPt_HI_] = &l3SingleMuPt_HI_pset;
  l3SingleMuEta_HI_pset = iConfig.getParameter<edm::ParameterSet>("l3SingleMuEta_HI");
  plotMap[&l3SingleMuEta_HI_] = &l3SingleMuEta_HI_pset;
  l3SingleMuPhi_HI_pset = iConfig.getParameter<edm::ParameterSet>("l3SingleMuPhi_HI");
  plotMap[&l3SingleMuPhi_HI_] = &l3SingleMuPhi_HI_pset;
  l3SingleMuHFPt_HI_pset = iConfig.getParameter<edm::ParameterSet>("l3SingleMuHFPt_HI");
  plotMap[&l3SingleMuHFPt_HI_] = &l3SingleMuHFPt_HI_pset;
  l3SingleMuHFEta_HI_pset = iConfig.getParameter<edm::ParameterSet>("l3SingleMuHFEta_HI");
  plotMap[&l3SingleMuHFEta_HI_] = &l3SingleMuHFEta_HI_pset;
  l3SingleMuHFPhi_HI_pset = iConfig.getParameter<edm::ParameterSet>("l3SingleMuHFPhi_HI");
  plotMap[&l3SingleMuHFPhi_HI_] = &l3SingleMuHFPhi_HI_pset;
  fullTrack24Pt_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack24Pt_HI");
  plotMap[&fullTrack24Pt_HI_] = &fullTrack24Pt_HI_pset;
  fullTrack24Eta_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack24Eta_HI");
  plotMap[&fullTrack24Eta_HI_] = &fullTrack24Eta_HI_pset;
  fullTrack24Phi_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack24Phi_HI");
  plotMap[&fullTrack24Phi_HI_] = &fullTrack24Phi_HI_pset;
  fullTrack24Centrality301Pt_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack24Centrality301Pt_HI");
  plotMap[&fullTrack24Centrality301Pt_HI_] = &fullTrack24Centrality301Pt_HI_pset;
  fullTrack24Centrality301Eta_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack24Centrality301Eta_HI");
  plotMap[&fullTrack24Centrality301Eta_HI_] = &fullTrack24Centrality301Eta_HI_pset;
  fullTrack24Centrality301Phi_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack24Centrality301Phi_HI");
  plotMap[&fullTrack24Centrality301Phi_HI_] = &fullTrack24Centrality301Phi_HI_pset;
  fullTrack34Pt_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack34Pt_HI");
  plotMap[&fullTrack34Pt_HI_] = &fullTrack34Pt_HI_pset;
  fullTrack34Eta_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack34Eta_HI");
  plotMap[&fullTrack34Eta_HI_] = &fullTrack34Eta_HI_pset;
  fullTrack34Phi_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack34Phi_HI");
  plotMap[&fullTrack34Phi_HI_] = &fullTrack34Phi_HI_pset;
  fullTrack34Centrality301Pt_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack34Centrality301Pt_HI");
  plotMap[&fullTrack34Centrality301Pt_HI_] = &fullTrack34Centrality301Pt_HI_pset;
  fullTrack34Centrality301Eta_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack34Centrality301Eta_HI");
  plotMap[&fullTrack34Centrality301Eta_HI_] = &fullTrack34Centrality301Eta_HI_pset;
  fullTrack34Centrality301Phi_HI_pset = iConfig.getParameter<edm::ParameterSet>("fullTrack34Centrality301Phi_HI");
  plotMap[&fullTrack34Centrality301Phi_HI_] = &fullTrack34Centrality301Phi_HI_pset;
  //pp ref
  caloJetRefPt_HI_pset = iConfig.getParameter<edm::ParameterSet>("caloJetRefPt_HI");
  plotMap[&caloJetRefPt_HI_] = &caloJetRefPt_HI_pset;
  caloJetRefEta_HI_pset = iConfig.getParameter<edm::ParameterSet>("caloJetRefEta_HI");
  plotMap[&caloJetRefEta_HI_] = &caloJetRefEta_HI_pset;
  caloJetRefPhi_HI_pset = iConfig.getParameter<edm::ParameterSet>("caloJetRefPhi_HI");
  plotMap[&caloJetRefPhi_HI_] = &caloJetRefPhi_HI_pset;
  dMesonRefTrackPt_HI_pset = iConfig.getParameter<edm::ParameterSet>("dMesonRefTrackPt_HI");
  plotMap[&dMesonRefTrackPt_HI_] = &dMesonRefTrackPt_HI_pset;
  dMesonRefTrackEta_HI_pset = iConfig.getParameter<edm::ParameterSet>("dMesonRefTrackEta_HI");
  plotMap[&dMesonRefTrackEta_HI_] = &dMesonRefTrackEta_HI_pset;
  dMesonRefTrackSystemMass_HI_pset = iConfig.getParameter<edm::ParameterSet>("dMesonRefTrackSystemMass_HI");
  plotMap[&dMesonRefTrackSystemMass_HI_] = &dMesonRefTrackSystemMass_HI_pset;
  dMesonRefTrackSystemPt_HI_pset = iConfig.getParameter<edm::ParameterSet>("dMesonRefTrackSystemPt_HI");
  plotMap[&dMesonRefTrackSystemPt_HI_] = &dMesonRefTrackSystemPt_HI_pset;



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
  triggerResultsToken_ = consumes<edm::TriggerResults>(edm::InputTag("TriggerResults","", processName_));
  aodTriggerToken_ = consumes<trigger::TriggerEvent>(edm::InputTag("hltTriggerSummaryAOD", "", processName_));
  lumiScalersToken_ = consumes<LumiScalersCollection>(edm::InputTag("hltScalersRawToDigi","",""));

}


HLTObjectMonitorHeavyIon::~HLTObjectMonitorHeavyIon()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
HLTObjectMonitorHeavyIon::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
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
       if (firedMap[plot->pathName]) continue;
       bool triggerAccept = false;
       const TriggerObjectCollection objects = aodTriggerEvent->getObjects();
       edm::InputTag moduleFilter;
       std::string pathName;
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
	   unsigned int moduleFilterIndex = aodTriggerEvent->filterIndex(moduleFilter);
	   
	   if (moduleFilterIndex+1 > aodTriggerEvent->sizeFilters()) return;
	   const Keys &keys = aodTriggerEvent->filterKeys( moduleFilterIndex );

	   ////////////////////////////////
	   ///
	   /// single-object plots
	   ///
	   ////////////////////////////////

	   if (pathName == caloJetPt_HI_.pathName)
	     {
	       for (const auto & key : keys)
		 {
		   caloJetPt_HI_.ME->Fill(objects[key].pt());
		   caloJetEta_HI_.ME->Fill(objects[key].eta());
		   caloJetPhi_HI_.ME->Fill(objects[key].phi());
		 }
	     }
	   else if (pathName == singlePhotonEta1p5Pt_HI_.pathName)
	     {
	       for (const auto & key : keys)
		 {
		   singlePhotonEta1p5Pt_HI_.ME->Fill(objects[key].pt());
		   singlePhotonEta1p5Eta_HI_.ME->Fill(objects[key].eta());
		   singlePhotonEta1p5Phi_HI_.ME->Fill(objects[key].phi());
		 }
	     }
	   else if (pathName == singlePhotonEta3p1Pt_HI_.pathName)
	     {
	       for (const auto & key : keys)
		 {
		   singlePhotonEta3p1Pt_HI_.ME->Fill(objects[key].pt());
		   singlePhotonEta3p1Eta_HI_.ME->Fill(objects[key].eta());
		   singlePhotonEta3p1Phi_HI_.ME->Fill(objects[key].phi());
		 }
	     }
	   else if (pathName == doublePhotonMass_HI_.pathName)
	     {
	       for (const auto & key : keys)
		 {
		   doublePhotonMass_HI_.ME->Fill(objects[key].mass());
		 }
	     }
	   else if (pathName == fullTrack12MinBiasHFPt_HI_.pathName)
	     {
	       for (const auto & key : keys)
		 {
		   fullTrack12MinBiasHFPt_HI_.ME->Fill(objects[key].pt());
		   fullTrack12MinBiasHFEta_HI_.ME->Fill(objects[key].eta());
		   fullTrack12MinBiasHFPhi_HI_.ME->Fill(objects[key].phi());
		 }
	     }
	   else if (pathName == fullTrack12Centrality010Pt_HI_.pathName)
	     {
	       for (const auto & key : keys)
		 {
		   fullTrack12Centrality010Pt_HI_.ME->Fill(objects[key].pt());
		   fullTrack12Centrality010Eta_HI_.ME->Fill(objects[key].eta());
		   fullTrack12Centrality010Phi_HI_.ME->Fill(objects[key].phi());
		 }
	     }
	   else if (pathName == fullTrack12Centrality301Pt_HI_.pathName)
	     {
	       for (const auto & key : keys)
		 {
		   fullTrack12Centrality301Pt_HI_.ME->Fill(objects[key].pt());
		   fullTrack12Centrality301Eta_HI_.ME->Fill(objects[key].eta());
		   fullTrack12Centrality301Phi_HI_.ME->Fill(objects[key].phi());
		 }
	     }
	   else if (pathName == fullTrack24Centrality301Pt_HI_.pathName)
	     {
	       for (const auto & key : keys)
		 {
		   fullTrack24Centrality301Pt_HI_.ME->Fill(objects[key].pt());
		   fullTrack24Centrality301Eta_HI_.ME->Fill(objects[key].eta());
		   fullTrack24Centrality301Phi_HI_.ME->Fill(objects[key].phi());
		 }
	     }
	   else if (pathName == fullTrack24Pt_HI_.pathName)
	     {
	       for (const auto & key : keys)
		 {
		   fullTrack24Pt_HI_.ME->Fill(objects[key].pt());
		   fullTrack24Eta_HI_.ME->Fill(objects[key].eta());
		   fullTrack24Phi_HI_.ME->Fill(objects[key].phi());
		 }
	     }
	   else if (pathName == fullTrack34Centrality301Pt_HI_.pathName)
	     {
	       for (const auto & key : keys)
		 {
		   fullTrack34Centrality301Pt_HI_.ME->Fill(objects[key].pt());
		   fullTrack34Centrality301Eta_HI_.ME->Fill(objects[key].eta());
		   fullTrack34Centrality301Phi_HI_.ME->Fill(objects[key].phi());
		 }
	     }
	   else if (pathName == fullTrack34Pt_HI_.pathName)
	     {
	       for (const auto & key : keys)
		 {
		   fullTrack34Pt_HI_.ME->Fill(objects[key].pt());
		   fullTrack34Eta_HI_.ME->Fill(objects[key].eta());
		   fullTrack34Phi_HI_.ME->Fill(objects[key].phi());
		 }
	     }
	   else if (pathName == dMesonTrackPt_HI_.pathName)
	     {
	       const double pi_mass(.105658);
	       unsigned int kCnt0 = 0;
	       for (const auto & key0: keys)
		 {
		   dMesonTrackPt_HI_.ME->Fill(objects[key0].pt());
		   dMesonTrackEta_HI_.ME->Fill(objects[key0].eta());

		   unsigned int kCnt1 = 0;
		   for (const auto & key1: keys)
		     {
		       if (key0 != key1 && kCnt1 > kCnt0) // avoid filling hists with same objs && avoid double counting separate objs
			 {
			   TLorentzVector tk1, tk2, diTk;
			   tk1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), pi_mass);
			   tk2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), pi_mass);
			   diTk = tk1+tk2;
			   dMesonTrackSystemMass_HI_.ME->Fill(diTk.M());
			   dMesonTrackSystemPt_HI_.ME->Fill(diTk.Pt());
			 }
		       kCnt1 +=1;
		     }
		   kCnt0 +=1;
		 }
	     }
	   else if (pathName == l1DoubleMuPt_HI_.pathName)
	     {
	       for (const auto & key : keys)
		 {
		   l1DoubleMuPt_HI_.ME->Fill(objects[key].pt());
		   l1DoubleMuEta_HI_.ME->Fill(objects[key].eta());
		   l1DoubleMuPhi_HI_.ME->Fill(objects[key].phi());
		 }
	     }
	   else if (pathName == l3DoubleMuPt_HI_.pathName)
	     {
	       for (const auto & key : keys)
		 {
		   l3DoubleMuPt_HI_.ME->Fill(objects[key].pt());
		   l3DoubleMuEta_HI_.ME->Fill(objects[key].eta());
		   l3DoubleMuPhi_HI_.ME->Fill(objects[key].phi());
		 }
	     }
	   else if (pathName == l2SingleMuPt_HI_.pathName)
	     {
	       for (const auto & key : keys)
		 {
		   l2SingleMuPt_HI_.ME->Fill(objects[key].pt());
		   l2SingleMuEta_HI_.ME->Fill(objects[key].eta());
		   l2SingleMuPhi_HI_.ME->Fill(objects[key].phi());
		 }
	     }
	   else if (pathName == l3SingleMuPt_HI_.pathName)
	     {
	       for (const auto & key : keys)
		 {
		   l3SingleMuPt_HI_.ME->Fill(objects[key].pt());
		   l3SingleMuEta_HI_.ME->Fill(objects[key].eta());
		   l3SingleMuPhi_HI_.ME->Fill(objects[key].phi());
		 }
	     }
	   else if (pathName == l3SingleMuHFPt_HI_.pathName)
	     {
	       for (const auto & key : keys)
		 {
		   l3SingleMuHFPt_HI_.ME->Fill(objects[key].pt());
		   l3SingleMuHFEta_HI_.ME->Fill(objects[key].eta());
		   l3SingleMuHFPhi_HI_.ME->Fill(objects[key].phi());
		 }
	     }
	   //pp ref paths
	   else if (pathName == caloJetRefPt_HI_.pathName) 
	     {
	       for (const auto & key : keys)
		 {
		   caloJetRefPt_HI_.ME->Fill(objects[key].pt());
		   caloJetRefEta_HI_.ME->Fill(objects[key].eta());
		   caloJetRefPhi_HI_.ME->Fill(objects[key].phi());
		 }
	     }
	   else if (pathName == dMesonRefTrackPt_HI_.pathName)
	     {

	       const double pi_mass(.105658);
	       unsigned int kCnt0 = 0;
	       for (const auto & key0: keys)
		 {
		   dMesonRefTrackPt_HI_.ME->Fill(objects[key0].pt());
		   dMesonRefTrackEta_HI_.ME->Fill(objects[key0].eta());

		   unsigned int kCnt1 = 0;
		   for (const auto & key1: keys)
		     {
		       if (key0 != key1 && kCnt1 > kCnt0) // avoid filling hists with same objs && avoid double counting separate objs
			 {
			   TLorentzVector tk1, tk2, diTk;
			   tk1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), pi_mass);
			   tk2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), pi_mass);
			   diTk = tk1+tk2;
			   dMesonRefTrackSystemMass_HI_.ME->Fill(diTk.M());
			   dMesonRefTrackSystemPt_HI_.ME->Fill(diTk.Pt());
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
HLTObjectMonitorHeavyIon::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
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
HLTObjectMonitorHeavyIon::endRun(edm::Run const&, edm::EventSetup const&)
{
  if (debugPrint) std::cout << "Calling endRun. " << std::endl;
}

void HLTObjectMonitorHeavyIon::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup)
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

double HLTObjectMonitorHeavyIon::get_wall_time()
{
  struct timeval time;
  if (gettimeofday(&time,NULL)) return 0;
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
HLTObjectMonitorHeavyIon::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
HLTObjectMonitorHeavyIon::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
// void
// HLTObjectMonitorHeavyIon::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
//   //The following says we do not know what parameters are allowed so do no validation
//   // Please change this to state exactly what you do use, even if it is no parameters
//   edm::ParameterSetDescription desc;
//   desc.setUnknown();
//   descriptions.addDefault(desc);
// }

//define this as a plug-in
DEFINE_FWK_MODULE(HLTObjectMonitorHeavyIon);
