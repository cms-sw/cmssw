// -*- C++ -*-
//
// Package:    DQM/HLTObjectMonitor
// Class:      HLTObjectMonitor
//
/**\class HLTObjectMonitor HLTObjectMonitor.cc DQM/HLTEvF/plugins/HLTObjectMonitor.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Charles Nicholas Mueller
//         Created:  Sun, 22 Mar 2015 22:29:00 GMT
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
using std::string;
using std::unordered_map;
using std::vector;

class HLTObjectMonitor : public DQMEDAnalyzer {
  struct hltPlot {
    MonitorElement* ME;
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
  explicit HLTObjectMonitor(const edm::ParameterSet&);
  ~HLTObjectMonitor() override;

  //      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker& i, edm::Run const&, edm::EventSetup const&) override;
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;
  vector<hltPlot*> plotList;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  double dxyFinder(double, double, edm::Handle<reco::RecoChargedCandidateCollection>, edm::Handle<reco::BeamSpot>);
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
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> chargedCandToken_;
  edm::EDGetTokenT<reco::JetTagCollection> csvCaloTagsToken_;
  edm::EDGetTokenT<reco::JetTagCollection> csvPfTagsToken_;
  edm::EDGetTokenT<vector<reco::CaloJet>> csvCaloJetsToken_;
  edm::EDGetTokenT<vector<reco::PFJet>> csvPfJetsToken_;

  //declare params
  edm::ParameterSet rsq_pset;
  edm::ParameterSet mr_pset;
  edm::ParameterSet alphaT_pset;
  edm::ParameterSet photonPt_pset;
  edm::ParameterSet photonEta_pset;
  edm::ParameterSet photonPhi_pset;
  edm::ParameterSet muonPt_pset;
  edm::ParameterSet muonEta_pset;
  edm::ParameterSet muonPhi_pset;
  edm::ParameterSet l2muonPt_pset;
  edm::ParameterSet l2muonEta_pset;
  edm::ParameterSet l2muonPhi_pset;
  edm::ParameterSet l2NoBPTXmuonPt_pset;
  edm::ParameterSet l2NoBPTXmuonEta_pset;
  edm::ParameterSet l2NoBPTXmuonPhi_pset;
  edm::ParameterSet electronPt_pset;
  edm::ParameterSet electronEta_pset;
  edm::ParameterSet electronPhi_pset;
  edm::ParameterSet jetPt_pset;
  edm::ParameterSet jetAK8Pt_pset;
  edm::ParameterSet jetAK8Mass_pset;
  edm::ParameterSet tauPt_pset;
  edm::ParameterSet diMuonLowMass_pset;
  edm::ParameterSet caloMetPt_pset;
  edm::ParameterSet caloMetPhi_pset;
  edm::ParameterSet pfMetPt_pset;
  edm::ParameterSet pfMetPhi_pset;
  edm::ParameterSet caloHtPt_pset;
  edm::ParameterSet pfHtPt_pset;
  edm::ParameterSet bJetPhi_pset;
  edm::ParameterSet bJetEta_pset;
  edm::ParameterSet bJetCSVCalo_pset;
  edm::ParameterSet bJetCSVPF_pset;
  edm::ParameterSet diMuonMass_pset;
  edm::ParameterSet pAL1DoubleMuZMass_pset;
  edm::ParameterSet pAL2DoubleMuZMass_pset;
  edm::ParameterSet pAL3DoubleMuZMass_pset;
  edm::ParameterSet diElecMass_pset;
  edm::ParameterSet muonDxy_pset;
  edm::ParameterSet wallTime_pset;

  string processName_;

  hltPlot rsq_;
  hltPlot mr_;
  hltPlot alphaT_;
  hltPlot photonPt_;
  hltPlot photonEta_;
  hltPlot photonPhi_;
  hltPlot muonPt_;
  hltPlot muonEta_;
  hltPlot muonPhi_;
  hltPlot l2muonPt_;
  hltPlot l2muonEta_;
  hltPlot l2muonPhi_;
  hltPlot l2NoBPTXmuonPt_;
  hltPlot l2NoBPTXmuonEta_;
  hltPlot l2NoBPTXmuonPhi_;
  hltPlot electronPt_;
  hltPlot electronEta_;
  hltPlot electronPhi_;
  hltPlot jetPt_;
  hltPlot jetAK8Pt_;
  hltPlot jetAK8Mass_;
  hltPlot tauPt_;
  hltPlot diMuonLowMass_;
  hltPlot caloMetPt_;
  hltPlot caloMetPhi_;
  hltPlot pfMetPt_;
  hltPlot pfMetPhi_;
  hltPlot caloHtPt_;
  hltPlot pfHtPt_;
  hltPlot bJetPhi_;
  hltPlot bJetEta_;
  hltPlot bJetCSVCalo_;
  hltPlot bJetCSVPF_;
  hltPlot diMuonMass_;
  hltPlot pAL1DoubleMuZMass_;
  hltPlot pAL2DoubleMuZMass_;
  hltPlot pAL3DoubleMuZMass_;
  hltPlot diElecMass_;
  hltPlot muonDxy_;
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
HLTObjectMonitor::HLTObjectMonitor(const edm::ParameterSet& iConfig)

{
  //now do what ever initialization is needed
  debugPrint = false;

  topDirectoryName = "HLT/ObjectMonitor";
  mainShifterFolder = topDirectoryName + "/MainShifter";
  backupFolder = topDirectoryName + "/Backup";

  //parse params
  processName_ = iConfig.getParameter<string>("processName");

  rsq_pset = iConfig.getParameter<edm::ParameterSet>("rsq");
  plotMap[&rsq_] = &rsq_pset;
  mr_pset = iConfig.getParameter<edm::ParameterSet>("mr");
  plotMap[&mr_] = &mr_pset;
  alphaT_pset = iConfig.getParameter<edm::ParameterSet>("alphaT");
  plotMap[&alphaT_] = &alphaT_pset;
  photonPt_pset = iConfig.getParameter<edm::ParameterSet>("photonPt");
  plotMap[&photonPt_] = &photonPt_pset;
  photonEta_pset = iConfig.getParameter<edm::ParameterSet>("photonEta");
  plotMap[&photonEta_] = &photonEta_pset;
  photonPhi_pset = iConfig.getParameter<edm::ParameterSet>("photonPhi");
  plotMap[&photonPhi_] = &photonPhi_pset;
  muonPt_pset = iConfig.getParameter<edm::ParameterSet>("muonPt");
  plotMap[&muonPt_] = &muonPt_pset;
  muonEta_pset = iConfig.getParameter<edm::ParameterSet>("muonEta");
  plotMap[&muonEta_] = &muonEta_pset;
  muonPhi_pset = iConfig.getParameter<edm::ParameterSet>("muonPhi");
  plotMap[&muonPhi_] = &muonPhi_pset;
  l2muonPt_pset = iConfig.getParameter<edm::ParameterSet>("l2muonPt");
  plotMap[&l2muonPt_] = &l2muonPt_pset;
  l2muonEta_pset = iConfig.getParameter<edm::ParameterSet>("l2muonEta");
  plotMap[&l2muonEta_] = &l2muonEta_pset;
  l2muonPhi_pset = iConfig.getParameter<edm::ParameterSet>("l2muonPhi");
  plotMap[&l2muonPhi_] = &l2muonPhi_pset;
  l2NoBPTXmuonPt_pset = iConfig.getParameter<edm::ParameterSet>("l2NoBPTXmuonPt");
  plotMap[&l2NoBPTXmuonPt_] = &l2NoBPTXmuonPt_pset;
  l2NoBPTXmuonEta_pset = iConfig.getParameter<edm::ParameterSet>("l2NoBPTXmuonEta");
  plotMap[&l2NoBPTXmuonEta_] = &l2NoBPTXmuonEta_pset;
  l2NoBPTXmuonPhi_pset = iConfig.getParameter<edm::ParameterSet>("l2NoBPTXmuonPhi");
  plotMap[&l2NoBPTXmuonPhi_] = &l2NoBPTXmuonPhi_pset;
  electronPt_pset = iConfig.getParameter<edm::ParameterSet>("electronPt");
  plotMap[&electronPt_] = &electronPt_pset;
  electronEta_pset = iConfig.getParameter<edm::ParameterSet>("electronEta");
  plotMap[&electronEta_] = &electronEta_pset;
  electronPhi_pset = iConfig.getParameter<edm::ParameterSet>("electronPhi");
  plotMap[&electronPhi_] = &electronPhi_pset;
  jetPt_pset = iConfig.getParameter<edm::ParameterSet>("jetPt");
  plotMap[&jetPt_] = &jetPt_pset;
  jetAK8Mass_pset = iConfig.getParameter<edm::ParameterSet>("jetAK8Mass");
  plotMap[&jetAK8Mass_] = &jetAK8Mass_pset;
  diMuonLowMass_pset = iConfig.getParameter<edm::ParameterSet>("diMuonLowMass");
  plotMap[&diMuonLowMass_] = &diMuonLowMass_pset;
  caloMetPt_pset = iConfig.getParameter<edm::ParameterSet>("caloMetPt");
  plotMap[&caloMetPt_] = &caloMetPt_pset;
  caloMetPhi_pset = iConfig.getParameter<edm::ParameterSet>("caloMetPhi");
  plotMap[&caloMetPhi_] = &caloMetPhi_pset;
  pfMetPt_pset = iConfig.getParameter<edm::ParameterSet>("pfMetPt");
  plotMap[&pfMetPt_] = &pfMetPt_pset;
  pfMetPhi_pset = iConfig.getParameter<edm::ParameterSet>("pfMetPhi");
  plotMap[&pfMetPhi_] = &pfMetPhi_pset;
  caloHtPt_pset = iConfig.getParameter<edm::ParameterSet>("caloHtPt");
  plotMap[&caloHtPt_] = &caloHtPt_pset;
  pfHtPt_pset = iConfig.getParameter<edm::ParameterSet>("pfHtPt");
  plotMap[&pfHtPt_] = &pfHtPt_pset;
  bJetPhi_pset = iConfig.getParameter<edm::ParameterSet>("bJetPhi");
  plotMap[&bJetPhi_] = &bJetPhi_pset;
  bJetEta_pset = iConfig.getParameter<edm::ParameterSet>("bJetEta");
  plotMap[&bJetEta_] = &bJetEta_pset;
  bJetCSVCalo_pset = iConfig.getParameter<edm::ParameterSet>("bJetCSVCalo");
  plotMap[&bJetCSVCalo_] = &bJetCSVCalo_pset;
  bJetCSVPF_pset = iConfig.getParameter<edm::ParameterSet>("bJetCSVPF");
  plotMap[&bJetCSVPF_] = &bJetCSVPF_pset;
  diMuonMass_pset = iConfig.getParameter<edm::ParameterSet>("diMuonMass");
  plotMap[&diMuonMass_] = &diMuonMass_pset;
  pAL1DoubleMuZMass_pset = iConfig.getParameter<edm::ParameterSet>("pAL1DoubleMuZMass");
  plotMap[&pAL1DoubleMuZMass_] = &pAL1DoubleMuZMass_pset;
  pAL2DoubleMuZMass_pset = iConfig.getParameter<edm::ParameterSet>("pAL2DoubleMuZMass");
  plotMap[&pAL2DoubleMuZMass_] = &pAL2DoubleMuZMass_pset;
  pAL3DoubleMuZMass_pset = iConfig.getParameter<edm::ParameterSet>("pAL3DoubleMuZMass");
  plotMap[&pAL3DoubleMuZMass_] = &pAL3DoubleMuZMass_pset;
  diElecMass_pset = iConfig.getParameter<edm::ParameterSet>("diElecMass");
  plotMap[&diElecMass_] = &diElecMass_pset;
  muonDxy_pset = iConfig.getParameter<edm::ParameterSet>("muonDxy");
  plotMap[&muonDxy_] = &muonDxy_pset;
  jetAK8Pt_pset = iConfig.getParameter<edm::ParameterSet>("jetAK8Pt");
  plotMap[&jetAK8Pt_] = &jetAK8Pt_pset;
  tauPt_pset = iConfig.getParameter<edm::ParameterSet>("tauPt");
  plotMap[&tauPt_] = &tauPt_pset;
  wallTime_pset = iConfig.getParameter<edm::ParameterSet>("wallTime");
  plotMap[&wallTime_] = &wallTime_pset;

  for (auto item = plotMap.begin(); item != plotMap.end(); item++) {
    (*item->first).pathName = (*item->second).getParameter<string>("pathName");
    (*item->first).moduleName = (*item->second).getParameter<string>("moduleName");
    (*item->first).nBins = (*item->second).getParameter<int>("NbinsX");
    (*item->first).xMin = (*item->second).getParameter<double>("Xmin");
    (*item->first).xMax = (*item->second).getParameter<double>("Xmax");
    (*item->first).xAxisLabel = (*item->second).getParameter<string>("axisLabel");
    (*item->first).plotLabel = (*item->second).getParameter<string>("plotLabel");
    (*item->first).displayInPrimary = (*item->second).getParameter<bool>("mainWorkspace");

    if ((*item->second).exists("pathName_OR")) {
      (*item->first).pathNameOR = (*item->second).getParameter<string>("pathName_OR");
    }
    if ((*item->second).exists("moduleName_OR")) {
      (*item->first).moduleNameOR = (*item->second).getParameter<string>("moduleName_OR");
    }

    plotList.push_back(item->first);
  }
  plotMap.clear();

  //set Token(s)
  triggerResultsToken_ = consumes<edm::TriggerResults>(edm::InputTag("TriggerResults", "", processName_));
  aodTriggerToken_ = consumes<trigger::TriggerEvent>(edm::InputTag("hltTriggerSummaryAOD", "", processName_));
  lumiScalersToken_ = consumes<LumiScalersCollection>(edm::InputTag("hltScalersRawToDigi", "", ""));
  beamSpotToken_ = consumes<reco::BeamSpot>(edm::InputTag("hltOnlineBeamSpot", "", processName_));
  chargedCandToken_ = consumes<vector<reco::RecoChargedCandidate>>(
      edm::InputTag("hltL3NoFiltersNoVtxMuonCandidates", "", processName_));
  csvCaloTagsToken_ =
      consumes<reco::JetTagCollection>(edm::InputTag("hltCombinedSecondaryVertexBJetTagsCalo", "", processName_));
  csvPfTagsToken_ =
      consumes<reco::JetTagCollection>(edm::InputTag("hltCombinedSecondaryVertexBJetTagsPF", "", processName_));
  csvCaloJetsToken_ =
      consumes<vector<reco::CaloJet>>(edm::InputTag("hltSelector8CentralJetsL1FastJet", "", processName_));
  csvPfJetsToken_ = consumes<vector<reco::PFJet>>(edm::InputTag("hltPFJetForBtag", "", processName_));
}

HLTObjectMonitor::~HLTObjectMonitor() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------
void HLTObjectMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  double start = get_wall_time();

  using namespace edm;

  if (debugPrint)
    std::cout << "Inside analyze(). " << std::endl;

  // access trigger results
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken_, triggerResults);
  if (!triggerResults.isValid())
    return;

  edm::Handle<trigger::TriggerEvent> aodTriggerEvent;
  iEvent.getByToken(aodTriggerToken_, aodTriggerEvent);
  if (!aodTriggerEvent.isValid())
    return;

  //reset everything to not accepted at beginning of each event
  unordered_map<string, bool> firedMap = acceptMap;
  for (auto plot : plotList)  //loop over paths
  {
    if (firedMap[plot->pathName])
      continue;
    bool triggerAccept = false;
    const TriggerObjectCollection objects = aodTriggerEvent->getObjects();
    edm::InputTag moduleFilter;
    std::string pathName;
    if (plot->pathIndex > 0 && triggerResults->accept(plot->pathIndex) && hltConfig_.saveTags(plot->moduleName)) {
      moduleFilter = edm::InputTag(plot->moduleName, "", processName_);
      pathName = plot->pathName;
      triggerAccept = true;
    } else if (plot->pathIndexOR > 0 && triggerResults->accept(plot->pathIndexOR) &&
               hltConfig_.saveTags(plot->moduleNameOR)) {
      if (firedMap[plot->pathNameOR])
        continue;
      moduleFilter = edm::InputTag(plot->moduleNameOR, "", processName_);
      pathName = plot->pathNameOR;
      triggerAccept = true;
    }

    if (triggerAccept) {
      unsigned int moduleFilterIndex = aodTriggerEvent->filterIndex(moduleFilter);

      if (moduleFilterIndex + 1 > aodTriggerEvent->sizeFilters())
        return;
      const Keys& keys = aodTriggerEvent->filterKeys(moduleFilterIndex);

      ////////////////////////////////
      ///
      /// single-object plots
      ///
      ////////////////////////////////

      //PFHT pt
      if (pathName == pfHtPt_.pathName) {
        for (const auto& key : keys)
          pfHtPt_.ME->Fill(objects[key].pt());
      }

      //jet pt
      else if (pathName == jetPt_.pathName) {
        for (const auto& key : keys)
          jetPt_.ME->Fill(objects[key].pt());
      }

      //photon pt + eta + phi (all use same path)
      else if (pathName == photonPt_.pathName) {
        for (const auto& key : keys) {
          photonPt_.ME->Fill(objects[key].pt());
          photonEta_.ME->Fill(objects[key].eta());
          photonPhi_.ME->Fill(objects[key].phi());
        }
      }

      //electron pt + eta + phi (all use same path)
      else if (pathName == electronPt_.pathName) {
        for (const auto& key : keys) {
          electronPt_.ME->Fill(objects[key].pt());
          electronEta_.ME->Fill(objects[key].eta());
          electronPhi_.ME->Fill(objects[key].phi());
        }
      }

      //muon pt + eta + phi (all use same path)
      else if (pathName == muonPt_.pathName) {
        for (const auto& key : keys) {
          muonPt_.ME->Fill(objects[key].pt());
          muonEta_.ME->Fill(objects[key].eta());
          muonPhi_.ME->Fill(objects[key].phi());
        }
      }

      //l2muon pt
      else if (pathName == l2muonPt_.pathName) {
        for (const auto& key : keys) {
          l2muonPt_.ME->Fill(objects[key].pt());
          l2muonEta_.ME->Fill(objects[key].eta());
          l2muonPhi_.ME->Fill(objects[key].phi());
        }
      }

      //l2NoBPTXmuon pt
      else if (pathName == l2NoBPTXmuonPt_.pathName) {
        for (const auto& key : keys) {
          l2NoBPTXmuonPt_.ME->Fill(objects[key].pt());
          l2NoBPTXmuonEta_.ME->Fill(objects[key].eta());
          l2NoBPTXmuonPhi_.ME->Fill(objects[key].phi());
        }
      }

      //Razor
      else if (pathName == mr_.pathName) {
        double onlineMR = 0, onlineRsq = 0;
        for (const auto& key : keys) {
          if (objects[key].id() == 0) {    //the MET object containing MR and Rsq will show up with ID = 0
            onlineMR = objects[key].px();  //razor variables stored in dummy reco::MET objects
            onlineRsq = objects[key].py();
          }
          mr_.ME->Fill(onlineMR);
          rsq_.ME->Fill(onlineRsq);
        }
      }

      //alphaT
      else if (pathName == alphaT_.pathName) {
        std::vector<ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>> alphaT_jets;
        for (const auto& key : keys) {
          ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>> JetLVec(
              objects[key].pt(), objects[key].eta(), objects[key].phi(), objects[key].mass());
          alphaT_jets.push_back(JetLVec);
        }

        float alphaT = AlphaT(alphaT_jets, false).value();
        alphaT_.ME->Fill(alphaT);
      }

      //tau pt
      else if (pathName == tauPt_.pathName) {
        for (const auto& key : keys)
          tauPt_.ME->Fill(objects[key].pt());
      }

      //caloMET pt+phi
      else if (pathName == caloMetPt_.pathName) {
        for (const auto& key : keys) {
          caloMetPt_.ME->Fill(objects[key].pt());
          caloMetPhi_.ME->Fill(objects[key].phi());
        }
      }

      //caloHT pt
      else if (pathName == caloHtPt_.pathName) {
        for (const auto& key : keys) {
          if (objects[key].id() == 89)
            caloHtPt_.ME->Fill(objects[key].pt());
        }
      }

      //jetAK8 pt + mass
      else if (pathName == jetAK8Pt_.pathName) {
        for (const auto& key : keys) {
          jetAK8Pt_.ME->Fill(objects[key].pt());
          jetAK8Mass_.ME->Fill(objects[key].mass());
        }
      }

      //PFMET pt + phi
      else if (pathName == pfMetPt_.pathName) {
        for (const auto& key : keys) {
          pfMetPt_.ME->Fill(objects[key].pt());
          pfMetPhi_.ME->Fill(objects[key].phi());
        }
      }

      // bjet eta + phi
      else if (pathName == bJetEta_.pathName || pathName == bJetEta_.pathNameOR) {
        for (const auto& key : keys) {
          bJetEta_.ME->Fill(objects[key].eta());
          bJetPhi_.ME->Fill(objects[key].phi());
        }
      }

      //b-tagging CSV information
      if (pathName == bJetCSVPF_.pathName) {
        edm::Handle<reco::JetTagCollection> csvPfTags;
        iEvent.getByToken(csvPfTagsToken_, csvPfTags);
        edm::Handle<vector<reco::PFJet>> csvPfJets;
        iEvent.getByToken(csvPfJetsToken_, csvPfJets);

        if (csvPfTags.isValid() && csvPfJets.isValid()) {
          for (auto iter = csvPfTags->begin(); iter != csvPfTags->end(); iter++)
            bJetCSVPF_.ME->Fill(iter->second);
        }
      }
      if (pathName == bJetCSVCalo_.pathName) {
        edm::Handle<reco::JetTagCollection> csvCaloTags;
        iEvent.getByToken(csvCaloTagsToken_, csvCaloTags);
        edm::Handle<vector<reco::CaloJet>> csvCaloJets;
        iEvent.getByToken(csvCaloJetsToken_, csvCaloJets);

        if (csvCaloTags.isValid() && csvCaloJets.isValid()) {
          for (auto iter = csvCaloTags->begin(); iter != csvCaloTags->end(); iter++)
            bJetCSVCalo_.ME->Fill(iter->second);
        }
      }

      //muon dxy(use an unique path)
      else if (pathName == muonDxy_.pathName) {
        edm::Handle<vector<reco::RecoChargedCandidate>> recoChargedCands;
        iEvent.getByToken(chargedCandToken_, recoChargedCands);
        edm::Handle<reco::BeamSpot> recoBeamSpot;
        iEvent.getByToken(beamSpotToken_, recoBeamSpot);
        double muon_dxy;

        if (recoChargedCands.isValid() && recoBeamSpot.isValid()) {
          for (const auto& key : keys) {
            muon_dxy = dxyFinder(objects[key].eta(), objects[key].phi(), recoChargedCands, recoBeamSpot);
            if (muon_dxy != -99.)
              muonDxy_.ME->Fill(muon_dxy);
          }
        }
      }

      // ////////////////////////////////
      // ///
      // /// double-object plots
      // ///
      // ////////////////////////////////

      //double muon low mass
      else if (pathName == diMuonLowMass_.pathName) {
        const double mu_mass(.105658);
        unsigned int kCnt0 = 0;
        for (const auto& key0 : keys) {
          unsigned int kCnt1 = 0;
          for (const auto& key1 : keys) {
            if (key0 != key1 &&
                kCnt1 > kCnt0)  // avoid filling hists with same objs && avoid double counting separate objs
            {
              if (abs(objects[key0].id()) == 13 &&
                  (objects[key0].id() + objects[key1].id() == 0))  // check muon id and dimuon charge
              {
                TLorentzVector mu1, mu2, dimu;
                mu1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), mu_mass);
                mu2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), mu_mass);
                dimu = mu1 + mu2;
                diMuonLowMass_.ME->Fill(dimu.M());
              }
            }
            kCnt1 += 1;
          }
          kCnt0 += 1;
        }
      }  //end double object plot

      else if (pathName == diMuonMass_.pathName || pathName == diMuonMass_.pathNameOR) {
        const double mu_mass(.105658);
        unsigned int kCnt0 = 0;
        for (const auto& key0 : keys) {
          unsigned int kCnt1 = 0;
          for (const auto& key1 : keys) {
            if (key0 != key1 &&
                kCnt1 > kCnt0)  // avoid filling hists with same objs && avoid double counting separate objs
            {
              if (abs(objects[key0].id()) == 13 &&
                  (objects[key0].id() + objects[key1].id() == 0))  // check muon id and dimuon charge
              {
                TLorentzVector mu1, mu2, dimu;
                mu1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), mu_mass);
                mu2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), mu_mass);
                dimu = mu1 + mu2;
                diMuonMass_.ME->Fill(dimu.M());
              }
            }
            kCnt1 += 1;
          }
          kCnt0 += 1;
        }
      }

      else if (pathName == pAL1DoubleMuZMass_.pathName) {
        const double mu_mass(.105658);
        unsigned int kCnt0 = 0;
        for (const auto& key0 : keys) {
          unsigned int kCnt1 = 0;
          for (const auto& key1 : keys) {
            if (key0 != key1 &&
                kCnt1 > kCnt0)  // avoid filling hists with same objs && avoid double counting separate objs
            {
              // if (abs(objects[key0].id()) == 13 && (objects[key0].id()+objects[key1].id()==0))  // id is not filled for l1 stage2 muons
              //  {
              TLorentzVector mu1, mu2, dimu;
              mu1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), mu_mass);
              mu2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), mu_mass);
              dimu = mu1 + mu2;
              if (dimu.M() > pAL1DoubleMuZMass_.xMin && dimu.M() < pAL1DoubleMuZMass_.xMax)
                pAL1DoubleMuZMass_.ME->Fill(dimu.M());
              //  }
            }
            kCnt1 += 1;
          }
          kCnt0 += 1;
        }
      }

      else if (pathName == pAL2DoubleMuZMass_.pathName) {
        const double mu_mass(.105658);
        unsigned int kCnt0 = 0;
        for (const auto& key0 : keys) {
          unsigned int kCnt1 = 0;
          for (const auto& key1 : keys) {
            if (key0 != key1 &&
                kCnt1 > kCnt0)  // avoid filling hists with same objs && avoid double counting separate objs
            {
              if (abs(objects[key0].id()) == 13 &&
                  (objects[key0].id() + objects[key1].id() == 0))  // check muon id and dimuon charge
              {
                TLorentzVector mu1, mu2, dimu;
                mu1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), mu_mass);
                mu2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), mu_mass);
                dimu = mu1 + mu2;
                if (dimu.M() > pAL2DoubleMuZMass_.xMin && dimu.M() < pAL2DoubleMuZMass_.xMax)
                  pAL2DoubleMuZMass_.ME->Fill(dimu.M());
              }
            }
            kCnt1 += 1;
          }
          kCnt0 += 1;
        }
      }

      else if (pathName == pAL3DoubleMuZMass_.pathName) {
        const double mu_mass(.105658);
        unsigned int kCnt0 = 0;
        for (const auto& key0 : keys) {
          unsigned int kCnt1 = 0;
          for (const auto& key1 : keys) {
            if (key0 != key1 &&
                kCnt1 > kCnt0)  // avoid filling hists with same objs && avoid double counting separate objs
            {
              if (abs(objects[key0].id()) == 13 &&
                  (objects[key0].id() + objects[key1].id() == 0))  // check muon id and dimuon charge
              {
                TLorentzVector mu1, mu2, dimu;
                mu1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), mu_mass);
                mu2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), mu_mass);
                dimu = mu1 + mu2;
                if (dimu.M() > pAL3DoubleMuZMass_.xMin && dimu.M() < pAL3DoubleMuZMass_.xMax)
                  pAL3DoubleMuZMass_.ME->Fill(dimu.M());
              }
            }
            kCnt1 += 1;
          }
          kCnt0 += 1;
        }
      }

      else if (pathName == diElecMass_.pathName) {
        unsigned int kCnt0 = 0;
        for (const auto& key0 : keys) {
          unsigned int kCnt1 = 0;
          for (const auto& key1 : keys) {
            if (key0 != key1 &&
                kCnt1 > kCnt0)  // avoid filling hists with same objs && avoid double counting separate objs
            {
              //                   if (abs(objects[key0].id()) == 11 && (objects[key0].id()+objects[key1].id()==0))  // id is not filled for electrons
              //                     {
              TLorentzVector el1, el2, diEl;
              el1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), 0);
              el2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), 0);
              diEl = el1 + el2;
              diElecMass_.ME->Fill(diEl.M());
              //                     }
            }
            kCnt1 += 1;
          }
          kCnt0 += 1;
        }
      }  //end double object plot

      firedMap[pathName] = true;
    }  //end if trigger accept
  }    //end loop over plots/paths

  //   sleep(1); //sleep for 1s, used to calibrate timing
  double end = get_wall_time();
  double wallTime = end - start;
  wallTime_.ME->Fill(wallTime);
}

// ------------ method called when starting to processes a run  ------------
void HLTObjectMonitor::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  if (debugPrint)
    std::cout << "Calling beginRun. " << std::endl;
  bool changed = true;
  if (hltConfig_.init(iRun, iSetup, processName_, changed)) {
    if (debugPrint)
      std::cout << "Extracting HLTconfig. " << std::endl;
  }

  //get path indicies from menu
  string pathName_noVersion;
  vector<string> triggerPaths = hltConfig_.triggerNames();

  for (const auto& pathName : triggerPaths) {
    pathName_noVersion = hltConfig_.removeVersion(pathName);
    for (auto plot : plotList) {
      if (plot->pathName == pathName_noVersion) {
        (*plot).pathIndex = hltConfig_.triggerIndex(pathName);
      } else if (plot->pathNameOR == pathName_noVersion) {
        (*plot).pathIndexOR = hltConfig_.triggerIndex(pathName);
      }
    }
  }
  vector<hltPlot*> plotList_temp;
  for (auto plot : plotList) {
    if (plot->pathIndex > 0 || plot->pathIndexOR > 0) {
      plotList_temp.push_back(plot);
      acceptMap[plot->pathName] = false;
      if (plot->pathIndexOR > 0)
        acceptMap[plot->pathNameOR] = false;
    }
  }
  //now re-assign plotList to contain only the plots with paths in the menu.
  plotList = plotList_temp;
  plotList_temp.clear();
}

// ------------ method called when ending the processing of a run  ------------

void HLTObjectMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) {
  ////////////////////////////////
  ///
  /// Main shifter workspace plots
  ///
  ////////////////////////////////

  //book wall time separately
  ibooker.setCurrentFolder(mainShifterFolder);
  wallTime_.ME =
      ibooker.book1D(wallTime_.plotLabel, wallTime_.pathName, wallTime_.nBins, wallTime_.xMin, wallTime_.xMax);
  wallTime_.ME->setAxisTitle(wallTime_.xAxisLabel);

  for (auto plot : plotList) {
    std::string display_pathNames = plot->pathName;
    if (!plot->pathNameOR.empty())
      display_pathNames = plot->pathName + " OR " + plot->pathNameOR;

    if (plot->displayInPrimary) {
      ibooker.setCurrentFolder(mainShifterFolder);
      (*plot).ME = ibooker.book1D(plot->plotLabel, display_pathNames.c_str(), plot->nBins, plot->xMin, plot->xMax);
      (*plot).ME->setAxisTitle(plot->xAxisLabel);
      //need to add OR statement
    } else {
      ibooker.setCurrentFolder(backupFolder);
      (*plot).ME = ibooker.book1D(plot->plotLabel, display_pathNames.c_str(), plot->nBins, plot->xMin, plot->xMax);
      (*plot).ME->setAxisTitle(plot->xAxisLabel);
    }
  }
}

double HLTObjectMonitor::dxyFinder(double eta,
                                   double phi,
                                   edm::Handle<reco::RecoChargedCandidateCollection> recoChargedCands,
                                   edm::Handle<reco::BeamSpot> recoBeamSpot) {
  double dxy = -99.;
  for (reco::RecoChargedCandidateCollection::const_iterator l3Muon = recoChargedCands->begin();
       l3Muon != recoChargedCands->end();
       l3Muon++) {
    if (deltaR(eta, phi, l3Muon->eta(), l3Muon->phi()) < 0.1) {
      dxy = (-(l3Muon->vx() - recoBeamSpot->x0()) * l3Muon->py() + (l3Muon->vy() - recoBeamSpot->y0()) * l3Muon->px()) /
            l3Muon->pt();
      break;
    }
  }
  return dxy;
}

double HLTObjectMonitor::get_wall_time() {
  struct timeval time;
  if (gettimeofday(&time, nullptr))
    return 0;
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
HLTObjectMonitor::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
HLTObjectMonitor::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
// void
// HLTObjectMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
//   //The following says we do not know what parameters are allowed so do no validation
//   // Please change this to state exactly what you do use, even if it is no parameters
//   edm::ParameterSetDescription desc;
//   desc.setUnknown();
//   descriptions.addDefault(desc);
// }

//define this as a plug-in
DEFINE_FWK_MODULE(HLTObjectMonitor);
