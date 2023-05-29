#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <TFile.h>
#include <TTree.h>
#include "TPRegexp.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/HcalCalibObjects/interface/HcalHBHEMuonVariables.h"

//#define EDM_ML_DEBUG

class HcalHBHENewMuonAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit HcalHBHENewMuonAnalyzer(const edm::ParameterSet&);
  ~HcalHBHENewMuonAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  const edm::InputTag labelHBHEMuonVar_;
  const int useRaw_;
  const int maxDepth_;
  int kount_;

  const edm::EDGetTokenT<HcalHBHEMuonVariablesCollection> tokHBHEMuonVar_;

  //////////////////////////////////////////////////////
  static const int depthMax_ = 7;
  TTree* tree_;
  unsigned int runNumber_, eventNumber_, lumiNumber_, bxNumber_;
  unsigned int goodVertex_;
  bool muon_is_good_, muon_global_, muon_tracker_;
  bool muon_is_tight_, muon_is_medium_;
  double ptGlob_, etaGlob_, phiGlob_, energyMuon_, pMuon_;
  float muon_trkKink_, muon_chi2LocalPosition_, muon_segComp_;
  int trackerLayer_, numPixelLayers_, tight_PixelHits_;
  bool innerTrack_, outerTrack_, globalTrack_;
  double chiTracker_, dxyTracker_, dzTracker_;
  double innerTrackpt_, innerTracketa_, innerTrackphi_;
  double tight_validFraction_, outerTrackChi_;
  double outerTrackPt_, outerTrackEta_, outerTrackPhi_;
  int outerTrackHits_, outerTrackRHits_;
  double globalTrckPt_, globalTrckEta_, globalTrckPhi_;
  int globalMuonHits_, matchedStat_;
  double chiGlobal_, tight_LongPara_, tight_TransImpara_;
  double isolationR04_, isolationR03_;
  double ecalEnergy_, hcalEnergy_, hoEnergy_;
  bool matchedId_, hcalHot_;
  double ecal3x3Energy_, hcal1x1Energy_;
  unsigned int ecalDetId_, hcalDetId_, ehcalDetId_;
  int hcal_ieta_, hcal_iphi_;
  double hcalDepthEnergy_[depthMax_];
  double hcalDepthActiveLength_[depthMax_];
  double hcalDepthEnergyHot_[depthMax_];
  double hcalDepthActiveLengthHot_[depthMax_];
  double hcalDepthChargeHot_[depthMax_];
  double hcalDepthChargeHotBG_[depthMax_];
  double hcalDepthEnergyCorr_[depthMax_];
  double hcalDepthEnergyHotCorr_[depthMax_];
  bool hcalDepthMatch_[depthMax_];
  bool hcalDepthMatchHot_[depthMax_];
  double hcalActiveLength_, hcalActiveLengthHot_;
  std::vector<std::string> all_triggers_;
  std::vector<int> hltresults_;
  ////////////////////////////////////////////////////////////
};

HcalHBHENewMuonAnalyzer::HcalHBHENewMuonAnalyzer(const edm::ParameterSet& iConfig)
    : labelHBHEMuonVar_(iConfig.getParameter<edm::InputTag>("hbheMuonLabel")),
      useRaw_(iConfig.getParameter<int>("useRaw")),
      maxDepth_(iConfig.getUntrackedParameter<int>("maxDepth", 7)),
      tokHBHEMuonVar_(consumes<HcalHBHEMuonVariablesCollection>(labelHBHEMuonVar_)) {
  usesResource(TFileService::kSharedResource);
  //now do what ever initialization is needed
  kount_ = 0;

  edm::LogVerbatim("HBHEMuon") << "Parameters read from config file \n\t maxDepth__ " << maxDepth_ << "\n\t Labels "
                               << labelHBHEMuonVar_;
}

//
// member functions
//

// ------------ method called for each event  ------------
void HcalHBHENewMuonAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  ++kount_;
  runNumber_ = iEvent.id().run();
  eventNumber_ = iEvent.id().event();
  lumiNumber_ = iEvent.id().luminosityBlock();
  bxNumber_ = iEvent.bunchCrossing();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HBHEMuon") << "Run " << runNumber_ << " Event " << eventNumber_ << " Lumi " << lumiNumber_ << " BX "
                               << bxNumber_ << std::endl;
#endif

  auto const& hbheMuonColl = iEvent.getHandle(tokHBHEMuonVar_);
  if (hbheMuonColl.isValid()) {
    for (const auto& itr : (*(hbheMuonColl.product()))) {
      goodVertex_ = itr.goodVertex_;
      muon_is_good_ = itr.muonGood_;
      muon_global_ = itr.muonGlobal_;
      muon_tracker_ = itr.muonTracker_;
      muon_is_tight_ = itr.muonTight_;
      muon_is_medium_ = itr.muonMedium_;
      ptGlob_ = itr.ptGlob_;
      etaGlob_ = itr.etaGlob_;
      phiGlob_ = itr.phiGlob_;
      energyMuon_ = itr.energyMuon_;
      pMuon_ = itr.pMuon_;
      muon_trkKink_ = itr.muonTrkKink_;
      muon_chi2LocalPosition_ = itr.muonChi2LocalPosition_;
      muon_segComp_ = itr.muonSegComp_;
      trackerLayer_ = itr.trackerLayer_;
      numPixelLayers_ = itr.numPixelLayers_;
      tight_PixelHits_ = itr.tightPixelHits_;
      innerTrack_ = itr.innerTrack_;
      outerTrack_ = itr.outerTrack_;
      globalTrack_ = itr.globalTrack_;
      chiTracker_ = itr.chiTracker_;
      dxyTracker_ = itr.dxyTracker_;
      dzTracker_ = itr.dzTracker_;
      innerTrackpt_ = itr.innerTrackPt_;
      innerTracketa_ = itr.innerTrackEta_;
      innerTrackphi_ = itr.innerTrackPhi_;
      tight_validFraction_ = itr.tightValidFraction_;
      outerTrackChi_ = itr.outerTrackChi_;
      outerTrackPt_ = itr.outerTrackPt_;
      outerTrackEta_ = itr.outerTrackEta_;
      outerTrackPhi_ = itr.outerTrackPhi_;
      outerTrackHits_ = itr.outerTrackHits_;
      outerTrackRHits_ = itr.outerTrackRHits_;
      globalTrckPt_ = itr.globalTrackPt_;
      globalTrckEta_ = itr.globalTrackEta_;
      globalTrckPhi_ = itr.globalTrackPhi_;
      globalMuonHits_ = itr.globalMuonHits_;
      matchedStat_ = itr.matchedStat_;
      chiGlobal_ = itr.chiGlobal_;
      tight_LongPara_ = itr.tightLongPara_;
      tight_TransImpara_ = itr.tightTransImpara_;
      isolationR04_ = itr.isolationR04_;
      isolationR03_ = itr.isolationR03_;
      ecalEnergy_ = itr.ecalEnergy_;
      hcalEnergy_ = itr.hcalEnergy_;
      hoEnergy_ = itr.hoEnergy_;
      matchedId_ = itr.matchedId_;
      hcalHot_ = itr.hcalHot_;
      ecal3x3Energy_ = itr.ecal3x3Energy_;
      ecalDetId_ = itr.ecalDetId_;
      hcalDetId_ = itr.hcalDetId_;
      ehcalDetId_ = itr.ehcalDetId_;
      hcal_ieta_ = itr.hcalIeta_;
      hcal_iphi_ = itr.hcalIphi_;
      if (useRaw_ == 1)
        hcal1x1Energy_ = itr.hcal1x1EnergyAux_;
      else if (useRaw_ == 2)
        hcal1x1Energy_ = itr.hcal1x1EnergyRaw_;
      else
        hcal1x1Energy_ = itr.hcal1x1Energy_;
      for (unsigned int i = 0; i < itr.hcalDepthEnergy_.size(); ++i) {
        hcalDepthActiveLength_[i] = itr.hcalDepthActiveLength_[i];
        hcalDepthActiveLengthHot_[i] = itr.hcalDepthActiveLengthHot_[i];
        if (useRaw_ == 1) {
          hcalDepthEnergy_[i] = itr.hcalDepthEnergyAux_[i];
          hcalDepthEnergyHot_[i] = itr.hcalDepthEnergyHotAux_[i];
          hcalDepthEnergyCorr_[i] = itr.hcalDepthEnergyCorrAux_[i];
          hcalDepthEnergyHotCorr_[i] = itr.hcalDepthEnergyHotCorrAux_[i];
          hcalDepthChargeHot_[i] = itr.hcalDepthChargeHotAux_[i];
          hcalDepthChargeHotBG_[i] = itr.hcalDepthChargeHotBGAux_[i];
        } else if (useRaw_ == 2) {
          hcalDepthEnergy_[i] = itr.hcalDepthEnergyRaw_[i];
          hcalDepthEnergyHot_[i] = itr.hcalDepthEnergyHotRaw_[i];
          hcalDepthEnergyCorr_[i] = itr.hcalDepthEnergyCorrRaw_[i];
          hcalDepthEnergyHotCorr_[i] = itr.hcalDepthEnergyHotCorrRaw_[i];
          hcalDepthChargeHot_[i] = itr.hcalDepthChargeHotRaw_[i];
          hcalDepthChargeHotBG_[i] = itr.hcalDepthChargeHotBGRaw_[i];
        } else {
          hcalDepthEnergy_[i] = itr.hcalDepthEnergy_[i];
          hcalDepthEnergyHot_[i] = itr.hcalDepthEnergyHot_[i];
          hcalDepthEnergyCorr_[i] = itr.hcalDepthEnergyCorr_[i];
          hcalDepthEnergyHotCorr_[i] = itr.hcalDepthEnergyHotCorr_[i];
          hcalDepthChargeHot_[i] = itr.hcalDepthChargeHot_[i];
          hcalDepthChargeHotBG_[i] = itr.hcalDepthChargeHotBG_[i];
        }
        hcalDepthMatch_[i] = itr.hcalDepthMatch_[i];
        hcalDepthMatchHot_[i] = itr.hcalDepthMatchHot_[i];
      }
      hcalActiveLength_ = itr.hcalActiveLength_;
      hcalActiveLengthHot_ = itr.hcalActiveLengthHot_;
      all_triggers_ = itr.allTriggers_;
      hltresults_ = itr.hltResults_;
      tree_->Fill();
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void HcalHBHENewMuonAnalyzer::beginJob() {
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("TREE", "TREE");
  tree_->Branch("Event_No", &eventNumber_);
  tree_->Branch("Run_No", &runNumber_);
  tree_->Branch("LumiNumber", &lumiNumber_);
  tree_->Branch("BXNumber", &bxNumber_);
  tree_->Branch("GoodVertex", &goodVertex_);
  tree_->Branch("PF_Muon", &muon_is_good_);
  tree_->Branch("Global_Muon", &muon_global_);
  tree_->Branch("Tracker_muon", &muon_tracker_);
  tree_->Branch("MuonIsTight", &muon_is_tight_);
  tree_->Branch("MuonIsMedium", &muon_is_medium_);
  tree_->Branch("pt_of_muon", &ptGlob_);
  tree_->Branch("eta_of_muon", &etaGlob_);
  tree_->Branch("phi_of_muon", &phiGlob_);
  tree_->Branch("energy_of_muon", &energyMuon_);
  tree_->Branch("p_of_muon", &pMuon_);
  tree_->Branch("muon_trkKink", &muon_trkKink_);
  tree_->Branch("muon_chi2LocalPosition", &muon_chi2LocalPosition_);
  tree_->Branch("muon_segComp", &muon_segComp_);

  tree_->Branch("TrackerLayer", &trackerLayer_);
  tree_->Branch("NumPixelLayers", &numPixelLayers_);
  tree_->Branch("InnerTrackPixelHits", &tight_PixelHits_);
  tree_->Branch("innerTrack", &innerTrack_);
  tree_->Branch("chiTracker", &chiTracker_);
  tree_->Branch("DxyTracker", &dxyTracker_);
  tree_->Branch("DzTracker", &dzTracker_);
  tree_->Branch("innerTrackpt", &innerTrackpt_);
  tree_->Branch("innerTracketa", &innerTracketa_);
  tree_->Branch("innerTrackphi", &innerTrackphi_);
  tree_->Branch("tight_validFraction", &tight_validFraction_);

  tree_->Branch("OuterTrack", &outerTrack_);
  tree_->Branch("OuterTrackChi", &outerTrackChi_);
  tree_->Branch("OuterTrackPt", &outerTrackPt_);
  tree_->Branch("OuterTrackEta", &outerTrackEta_);
  tree_->Branch("OuterTrackPhi", &outerTrackPhi_);
  tree_->Branch("OuterTrackHits", &outerTrackHits_);
  tree_->Branch("OuterTrackRHits", &outerTrackRHits_);

  tree_->Branch("GlobalTrack", &globalTrack_);
  tree_->Branch("GlobalTrckPt", &globalTrckPt_);
  tree_->Branch("GlobalTrckEta", &globalTrckEta_);
  tree_->Branch("GlobalTrckPhi", &globalTrckPhi_);
  tree_->Branch("Global_Muon_Hits", &globalMuonHits_);
  tree_->Branch("MatchedStations", &matchedStat_);
  tree_->Branch("GlobTrack_Chi", &chiGlobal_);
  tree_->Branch("Tight_LongitudinalImpactparameter", &tight_LongPara_);
  tree_->Branch("Tight_TransImpactparameter", &tight_TransImpara_);

  tree_->Branch("IsolationR04", &isolationR04_);
  tree_->Branch("IsolationR03", &isolationR03_);
  tree_->Branch("ecal_3into3", &ecalEnergy_);
  tree_->Branch("hcal_3into3", &hcalEnergy_);
  tree_->Branch("tracker_3into3", &hoEnergy_);

  tree_->Branch("matchedId", &matchedId_);
  tree_->Branch("hcal_cellHot", &hcalHot_);

  tree_->Branch("ecal_3x3", &ecal3x3Energy_);
  tree_->Branch("hcal_1x1", &hcal1x1Energy_);
  tree_->Branch("ecal_detID", &ecalDetId_);
  tree_->Branch("hcal_detID", &hcalDetId_);
  tree_->Branch("ehcal_detID", &ehcalDetId_);
  tree_->Branch("hcal_ieta", &hcal_ieta_);
  tree_->Branch("hcal_iphi", &hcal_iphi_);

  char name[100];
  for (int k = 0; k < maxDepth_; ++k) {
    sprintf(name, "hcal_edepth%d", (k + 1));
    tree_->Branch(name, &hcalDepthEnergy_[k]);
    sprintf(name, "hcal_activeL%d", (k + 1));
    tree_->Branch(name, &hcalDepthActiveLength_[k]);
    sprintf(name, "hcal_edepthHot%d", (k + 1));
    tree_->Branch(name, &hcalDepthEnergyHot_[k]);
    sprintf(name, "hcal_activeHotL%d", (k + 1));
    tree_->Branch(name, &hcalDepthActiveLengthHot_[k]);
    sprintf(name, "hcal_cdepthHot%d", (k + 1));
    tree_->Branch(name, &hcalDepthChargeHot_[k]);
    sprintf(name, "hcal_cdepthHotBG%d", (k + 1));
    tree_->Branch(name, &hcalDepthChargeHotBG_[k]);
    sprintf(name, "hcal_edepthCorrect%d", (k + 1));
    tree_->Branch(name, &hcalDepthEnergyCorr_[k]);
    sprintf(name, "hcal_edepthHotCorrect%d", (k + 1));
    tree_->Branch(name, &hcalDepthEnergyHotCorr_[k]);
    sprintf(name, "hcal_depthMatch%d", (k + 1));
    tree_->Branch(name, &hcalDepthMatch_[k]);
    sprintf(name, "hcal_depthMatchHot%d", (k + 1));
    tree_->Branch(name, &hcalDepthMatchHot_[k]);
  }

  tree_->Branch("activeLength", &hcalActiveLength_);
  tree_->Branch("activeLengthHot", &hcalActiveLengthHot_);

  tree_->Branch("hltresults", &hltresults_);
  tree_->Branch("all_triggers", &all_triggers_);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HcalHBHENewMuonAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("hbheMuonLabel", edm::InputTag("alcaHcalHBHEMuonProducer", "hbheMuon"));
  desc.add<int>("useRaw", 0);
  desc.addUntracked<int>("maxDepth", 7);
  descriptions.add("hcalHBHEMuonAnalysis", desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HcalHBHENewMuonAnalyzer);
