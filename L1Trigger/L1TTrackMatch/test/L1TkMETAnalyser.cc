#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/L1TCorrelator/interface/TkEtMiss.h"
#include "DataFormats/L1TCorrelator/interface/TkEtMissFwd.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/Vertex.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "L1Trigger/L1TTrackMatch/interface/L1TkEtMissEmuAlgo.h"
#include "TEfficiency.h"
#include "TGraphAsymmErrors.h"
#include "TGraphErrors.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TPad.h"
#include "TProfile.h"
#include "TTree.h"

using namespace std;

class L1TkMETAnalyser : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit L1TkMETAnalyser(const edm::ParameterSet& iConfig);
  ~L1TkMETAnalyser() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void endJob() override;

  edm::ParameterSet config;

  edm::InputTag TrackMETSimInputTag;
  edm::InputTag TrackMETEmuInputTag;
  edm::InputTag TrackMETHWInputTag;

  edm::EDGetTokenT<std::vector<l1t::TkEtMiss>> TrackMETSimToken_;
  edm::EDGetTokenT<std::vector<l1t::EtSum>> TrackMETEmuToken_;
  edm::EDGetTokenT<std::vector<l1t::EtSum>> TrackMETHWToken_;

  float EtScale;
  float EtphiScale;

  bool available_;
  TTree* eventTree;

  std::vector<float>* m_SimMET;
  std::vector<float>* m_EmuMET;
  std::vector<float>* m_HwMET;

  std::vector<float>* m_SimMETphi;
  std::vector<float>* m_EmuMETphi;
  std::vector<float>* m_HwMETphi;

  std::vector<float>* m_SimNtrk;
  std::vector<float>* m_EmuNtrk;
  std::vector<float>* m_HwNtrk;

  bool HW_analysis_;

  TH1F* hisTkSimMET_;
  TH1F* hisTkEmuMET_;
  TH1F* hisTkHWMET_;

  TH1F* hisTkSimPhi_;
  TH1F* hisTkEmuPhi_;
  TH1F* hisTkHWPhi_;

  TH1F* hisTkSimNtrk_;
  TH1F* hisTkEmuNtrk_;
  TH1F* hisTkHWNtrk_;

  TH1F* hisMETResidual_;
  TH1F* hisPhiResidual_;
  TH1F* hisNtrkResidual_;
};

L1TkMETAnalyser::L1TkMETAnalyser(edm::ParameterSet const& iConfig) : config(iConfig) {
  HW_analysis_ = iConfig.getParameter<bool>("HW_Analysis");

  TrackMETSimInputTag = iConfig.getParameter<edm::InputTag>("TrackMETInputTag");
  TrackMETEmuInputTag = iConfig.getParameter<edm::InputTag>("TrackMETEmuInputTag");
  if (HW_analysis_) {
    TrackMETHWInputTag = iConfig.getParameter<edm::InputTag>("TrackMETHWInputTag");
  }
  TrackMETSimToken_ = consumes<std::vector<l1t::TkEtMiss>>(TrackMETSimInputTag);
  TrackMETEmuToken_ = consumes<std::vector<l1t::EtSum>>(TrackMETEmuInputTag);
  if (HW_analysis_) {
    TrackMETHWToken_ = consumes<std::vector<l1t::EtSum>>(TrackMETHWInputTag);
  }
  usesResource(TFileService::kSharedResource);
}

void L1TkMETAnalyser::beginJob() {
  edm::Service<TFileService> fs;
  TFileDirectory inputDir = fs->mkdir("TkMETAnalysis");
  available_ = fs.isAvailable();
  if (not available_)
    return;  // No ROOT file open.

  m_SimMET = new std::vector<float>;
  m_EmuMET = new std::vector<float>;
  m_HwMET = new std::vector<float>;

  m_SimMETphi = new std::vector<float>;
  m_EmuMETphi = new std::vector<float>;
  m_HwMETphi = new std::vector<float>;

  m_SimNtrk = new std::vector<float>;
  m_EmuNtrk = new std::vector<float>;
  m_HwNtrk = new std::vector<float>;

  eventTree = fs->make<TTree>("eventTree", "Event tree");

  eventTree->Branch("SimMET", &m_SimMET);
  eventTree->Branch("EmuMET", &m_EmuMET);
  eventTree->Branch("HwMET", &m_HwMET);

  eventTree->Branch("SimMETphi", &m_SimMETphi);
  eventTree->Branch("EmuMETphi", &m_EmuMETphi);
  eventTree->Branch("HwMETphi", &m_HwMETphi);

  eventTree->Branch("SimNtrk", &m_SimNtrk);
  eventTree->Branch("EmuNtrk", &m_EmuNtrk);
  eventTree->Branch("HwNtrk", &m_HwNtrk);

  hisTkSimMET_ = inputDir.make<TH1F>("hisTkSimMET_", "sim TkMET [GeV]", 101, 0, 500);
  hisTkEmuMET_ = inputDir.make<TH1F>("hisTkEmuMET_", "emu TkMET [GeV]", 101, 0, 500);

  hisTkSimPhi_ = inputDir.make<TH1F>("hisTkSimPhi_", "sim phi [rad]", 101, -M_PI, M_PI);
  hisTkEmuPhi_ = inputDir.make<TH1F>("hisTkEmuPhi_", "emu phi [rad]", 101, -M_PI, M_PI);

  hisTkSimNtrk_ = inputDir.make<TH1F>("hisTkSimNtrk_", "sim ntrks", 101, 0, 256);
  hisTkEmuNtrk_ = inputDir.make<TH1F>("hisTkEmuNtrk_", "emu ntrks", 101, 0, 256);

  if (!HW_analysis_) {
    hisMETResidual_ = inputDir.make<TH1F>("hisMETResidual_", "sim - emu TkMET [GeV]", 101, -100, 100);
    hisPhiResidual_ = inputDir.make<TH1F>("hisPhiResidual_", "sim - emu phi [rad]", 101, -1, 1);
    hisNtrkResidual_ = inputDir.make<TH1F>("hisNtrkResidual_", "sim - emu ntrks", 101, -10, 10);
  }

  if (HW_analysis_) {
    hisTkHWMET_ = inputDir.make<TH1F>("hisTkHWMET_", "hw TkMET [GeV]", 101, 0, 500);
    hisTkHWPhi_ = inputDir.make<TH1F>("hisTkHWPhi_", "hw phi [rad]", 101, -M_PI, M_PI);
    hisTkHWNtrk_ = inputDir.make<TH1F>("hisTkEmuNtrk_", "hw ntrks", 101, 0, 256);

    hisMETResidual_ = inputDir.make<TH1F>("hisMETResidual_", "emu - hw TkMET [GeV]", 101, -100, 100);
    hisPhiResidual_ = inputDir.make<TH1F>("hisPhiResidual_", "emu - hw phi [rad]", 101, -1, 1);
    hisNtrkResidual_ = inputDir.make<TH1F>("hisNtrkResidual_", "emu - hw ntrks", 101, -10, 10);
  }
}

void L1TkMETAnalyser::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (not available_)
    return;  // No ROOT file open.

  edm::Handle<std::vector<l1t::TkEtMiss>> L1TkMETSimHandle;
  edm::Handle<std::vector<l1t::EtSum>> L1TkMETEmuHandle;

  iEvent.getByToken(TrackMETSimToken_, L1TkMETSimHandle);
  iEvent.getByToken(TrackMETEmuToken_, L1TkMETEmuHandle);

  m_SimMET->clear();
  m_EmuMET->clear();
  m_HwMET->clear();
  m_SimMETphi->clear();
  m_EmuMETphi->clear();
  m_HwMETphi->clear();
  m_SimNtrk->clear();
  m_EmuNtrk->clear();
  m_HwNtrk->clear();

  float SimEtmiss = L1TkMETSimHandle->begin()->etMiss();
  float EmuEtmiss = L1TkMETEmuHandle->begin()->hwPt() * l1tmetemu::kStepMETwordEt;

  float SimEtPhi = L1TkMETSimHandle->begin()->etPhi();
  float EmuEtPhi = L1TkMETEmuHandle->begin()->hwPhi() * l1tmetemu::kStepMETwordPhi;

  int SimEtNtrk = L1TkMETSimHandle->begin()->etQual();
  int EmuEtNtrk = L1TkMETEmuHandle->begin()->hwQual();

  if (!HW_analysis_) {
    hisMETResidual_->Fill(EmuEtmiss - SimEtmiss);
    hisPhiResidual_->Fill(EmuEtPhi - SimEtPhi);
    hisNtrkResidual_->Fill(EmuEtNtrk - SimEtNtrk);
  }

  m_SimMET->push_back(SimEtmiss);
  m_EmuMET->push_back(EmuEtmiss);
  m_SimMETphi->push_back(SimEtPhi);
  m_EmuMETphi->push_back(EmuEtPhi);
  m_SimNtrk->push_back(SimEtNtrk);
  m_EmuNtrk->push_back(EmuEtNtrk);

  hisTkSimMET_->Fill(SimEtmiss);
  hisTkEmuMET_->Fill(EmuEtmiss);

  hisTkSimPhi_->Fill(SimEtPhi);
  hisTkEmuPhi_->Fill(EmuEtPhi);

  hisTkSimNtrk_->Fill(SimEtNtrk);
  hisTkEmuNtrk_->Fill(EmuEtNtrk);

  if (HW_analysis_) {
    edm::Handle<std::vector<l1t::EtSum>> L1TkMETHWHandle;
    iEvent.getByToken(TrackMETHWToken_, L1TkMETHWHandle);
    float HWEtmiss = L1TkMETHWHandle->begin()->hwPt();
    float HWEtPhi = L1TkMETHWHandle->begin()->hwPhi();
    int HWEtNtrk = L1TkMETHWHandle->begin()->hwQual();

    hisTkHWMET_->Fill(HWEtmiss);
    hisTkHWPhi_->Fill(HWEtPhi);
    hisTkHWNtrk_->Fill(HWEtNtrk);

    hisMETResidual_->Fill(EmuEtmiss - HWEtmiss);
    hisPhiResidual_->Fill(EmuEtPhi - HWEtPhi);
    hisNtrkResidual_->Fill(EmuEtNtrk - HWEtNtrk);

    m_HwMET->push_back(HWEtmiss);
    m_HwMETphi->push_back(HWEtPhi);
    m_HwNtrk->push_back(HWEtNtrk);
  }
  eventTree->Fill();
}

//////////
// END JOB
void L1TkMETAnalyser::endJob() {
  // things to be done at the exit of the event Loop
  edm::LogInfo("L1TkMETAnalyser") << "==================== TkMET RECONSTRUCTION ======================\n"
                                  << "MET Residual Bias: " << hisMETResidual_->GetMean() << " GeV\n"
                                  << "MET Resolution: " << hisMETResidual_->GetStdDev() << " GeV\n"
                                  << "Phi Residual Bias: " << hisPhiResidual_->GetMean() << " rad\n"
                                  << "Phi Resolution: " << hisPhiResidual_->GetStdDev() << " rad\n"
                                  << "NTrk Residual Bias: " << hisNtrkResidual_->GetMean() << " Tracks\n"
                                  << "Ntrk Resolution: " << hisNtrkResidual_->GetStdDev() << " Tracks\n";
}

void L1TkMETAnalyser::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

///////////////////////////
// DEFINE THIS AS A PLUG-IN
DEFINE_FWK_MODULE(L1TkMETAnalyser);
