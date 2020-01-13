#include <string>
#include <vector>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "CommonTools/TriggerUtils/interface/PrescaleWeightProvider.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"

class BPHMonitor : public DQMEDAnalyzer, public TriggerDQMBase {
public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  BPHMonitor(const edm::ParameterSet&);
  ~BPHMonitor() throw() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

  template <typename T>
  bool matchToTrigger(const std::string& theTriggerName, T t);

  double Prescale(const std::string num,
                  const std::string den,
                  edm::Event const& iEvent,
                  edm::EventSetup const& iSetup,
                  HLTPrescaleProvider* hltPrescale_);

private:
  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  edm::InputTag muoInputTag_;
  edm::InputTag bsInputTag_;
  edm::InputTag trInputTag_;
  edm::InputTag phInputTag_;
  edm::InputTag vtxInputTag_;

  edm::EDGetTokenT<reco::MuonCollection> muoToken_;
  edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  edm::EDGetTokenT<reco::TrackCollection> trToken_;
  edm::EDGetTokenT<reco::PhotonCollection> phToken_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;

  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;

  std::vector<double> pt_variable_binning_;
  std::vector<double> dMu_pt_variable_binning_;
  std::vector<double> prob_variable_binning_;
  MEbinning phi_binning_;
  MEbinning eta_binning_;
  MEbinning d0_binning_;
  MEbinning z0_binning_;
  MEbinning dR_binning_;
  MEbinning mass_binning_;
  MEbinning Bmass_binning_;
  MEbinning dca_binning_;
  MEbinning ds_binning_;
  MEbinning cos_binning_;

  ObjME muPhi_;
  ObjME muEta_;
  ObjME muPt_;
  ObjME mud0_;
  ObjME muz0_;

  ObjME mu1Phi_;
  ObjME mu1Eta_;
  ObjME mu1Pt_;
  ObjME mu1d0_;
  ObjME mu1z0_;
  ObjME mu2Phi_;
  ObjME mu2Eta_;
  ObjME mu2Pt_;
  ObjME mu2d0_;
  ObjME mu2z0_;
  ObjME mu3Phi_;
  ObjME mu3Eta_;
  ObjME mu3Pt_;
  ObjME mu3d0_;
  ObjME mu3z0_;

  ObjME phPhi_;
  ObjME phEta_;
  ObjME phPt_;
  ObjME DiMuPhi_;
  ObjME DiMuEta_;
  ObjME DiMuPt_;
  ObjME DiMuPVcos_;
  ObjME DiMuProb_;
  ObjME DiMuDS_;
  ObjME DiMuDCA_;
  ObjME DiMuMass_;
  ObjME BMass_;
  ObjME DiMudR_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  HLTPrescaleProvider* hltPrescale_;

  StringCutObjectSelector<reco::Muon, true> muoSelection_;
  StringCutObjectSelector<reco::Muon, true> muoSelection_ref;
  StringCutObjectSelector<reco::Muon, true> muoSelection_tag;
  StringCutObjectSelector<reco::Muon, true> muoSelection_probe;

  int nmuons_;
  bool tnp_;
  int L3_;
  int ptCut_;
  int displaced_;
  int trOrMu_;
  int Jpsi_;
  int Upsilon_;
  int enum_;
  int seagull_;
  double maxmass_;
  double minmass_;
  double maxmassJpsi;
  double minmassJpsi;
  double maxmassUpsilon;
  double minmassUpsilon;
  double maxmassTkTk;
  double minmassTkTk;
  double maxmassJpsiTk;
  double minmassJpsiTk;
  double kaon_mass;
  double mu_mass;
  double min_dR;
  double max_dR;

  double minprob;
  double mincos;
  double minDS;
  edm::EDGetTokenT<edm::TriggerResults> hltTrigResTag_;
  edm::InputTag hltInputTag_1;
  edm::EDGetTokenT<trigger::TriggerEvent> hltInputTag_;
  std::vector<std::string> hltpaths_num;
  std::vector<std::string> hltpaths_den;
  StringCutObjectSelector<reco::Track, true> trSelection_;
  StringCutObjectSelector<reco::Track, true> trSelection_ref;
  StringCutObjectSelector<reco::Candidate::LorentzVector, true> DMSelection_ref;

  edm::Handle<trigger::TriggerEvent> handleTriggerEvent;

  HLTConfigProvider hltConfig_;
  edm::Handle<edm::TriggerResults> HLTR;
  std::string getTriggerName(std::string partialName);
};

BPHMonitor::BPHMonitor(const edm::ParameterSet& iConfig)
    : folderName_(iConfig.getParameter<std::string>("FolderName")),
      requireValidHLTPaths_(iConfig.getParameter<bool>("requireValidHLTPaths")),
      hltPathsAreValid_(false),
      muoInputTag_(iConfig.getParameter<edm::InputTag>("muons")),
      bsInputTag_(iConfig.getParameter<edm::InputTag>("beamSpot")),
      trInputTag_(iConfig.getParameter<edm::InputTag>("tracks")),
      phInputTag_(iConfig.getParameter<edm::InputTag>("photons")),
      vtxInputTag_(iConfig.getParameter<edm::InputTag>("offlinePVs")),
      muoToken_(mayConsume<reco::MuonCollection>(muoInputTag_)),
      bsToken_(mayConsume<reco::BeamSpot>(bsInputTag_)),
      trToken_(mayConsume<reco::TrackCollection>(trInputTag_)),
      phToken_(mayConsume<reco::PhotonCollection>(phInputTag_)),
      vtxToken_(mayConsume<reco::VertexCollection>(vtxInputTag_)),
      pt_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double>>("ptBinning")),
      dMu_pt_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double>>("dMuPtBinning")),
      prob_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double>>("probBinning")),
      phi_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("phiPSet"))),
      eta_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("etaPSet"))),
      d0_binning_(
          getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("d0PSet"))),
      z0_binning_(
          getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("z0PSet"))),
      dR_binning_(
          getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("dRPSet"))),
      mass_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("massPSet"))),
      Bmass_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("BmassPSet"))),
      dca_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("dcaPSet"))),
      ds_binning_(
          getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("dsPSet"))),
      cos_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("cosPSet"))),
      num_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"), consumesCollector(), *this)),
      den_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"), consumesCollector(), *this)),
      hltPrescale_(new HLTPrescaleProvider(iConfig, consumesCollector(), *this)),
      muoSelection_(iConfig.getParameter<std::string>("muoSelection")),
      muoSelection_ref(iConfig.getParameter<std::string>("muoSelection_ref")),
      muoSelection_tag(iConfig.getParameter<std::string>("muoSelection_tag")),
      muoSelection_probe(iConfig.getParameter<std::string>("muoSelection_probe")),
      nmuons_(iConfig.getParameter<int>("nmuons")),
      tnp_(iConfig.getParameter<bool>("tnp")),
      L3_(iConfig.getParameter<int>("L3")),
      ptCut_(iConfig.getParameter<int>("ptCut")),
      displaced_(iConfig.getParameter<int>("displaced")),
      trOrMu_(iConfig.getParameter<int>("trOrMu")),
      Jpsi_(iConfig.getParameter<int>("Jpsi")),
      Upsilon_(iConfig.getParameter<int>("Upsilon"))  // if ==1 path with Upsilon constraint
      ,
      enum_(iConfig.getParameter<int>("enum")),
      seagull_(iConfig.getParameter<int>("seagull")),
      maxmass_(iConfig.getParameter<double>("maxmass")),
      minmass_(iConfig.getParameter<double>("minmass")),
      maxmassJpsi(iConfig.getParameter<double>("maxmassJpsi")),
      minmassJpsi(iConfig.getParameter<double>("minmassJpsi")),
      maxmassUpsilon(iConfig.getParameter<double>("maxmassUpsilon")),
      minmassUpsilon(iConfig.getParameter<double>("minmassUpsilon")),
      maxmassTkTk(iConfig.getParameter<double>("maxmassTkTk")),
      minmassTkTk(iConfig.getParameter<double>("minmassTkTk")),
      maxmassJpsiTk(iConfig.getParameter<double>("maxmassJpsiTk")),
      minmassJpsiTk(iConfig.getParameter<double>("minmassJpsiTk")),
      kaon_mass(iConfig.getParameter<double>("kaon_mass")),
      mu_mass(iConfig.getParameter<double>("mu_mass")),
      min_dR(iConfig.getParameter<double>("min_dR")),
      max_dR(iConfig.getParameter<double>("max_dR")),
      minprob(iConfig.getParameter<double>("minprob")),
      mincos(iConfig.getParameter<double>("mincos")),
      minDS(iConfig.getParameter<double>("minDS")),
      hltInputTag_1(iConfig.getParameter<edm::InputTag>("hltTriggerSummaryAOD")),
      hltInputTag_(mayConsume<trigger::TriggerEvent>(iConfig.getParameter<edm::InputTag>("hltTriggerSummaryAOD"))),
      hltpaths_num(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet")
                       .getParameter<std::vector<std::string>>("hltPaths")),
      hltpaths_den(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet")
                       .getParameter<std::vector<std::string>>("hltPaths")),
      trSelection_(iConfig.getParameter<std::string>("muoSelection")),
      trSelection_ref(iConfig.getParameter<std::string>("trSelection_ref")),
      DMSelection_ref(iConfig.getParameter<std::string>("DMSelection_ref")) {
  if (!tnp_) {
    magneticFieldToken_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();
  }
}

BPHMonitor::~BPHMonitor() throw() {
  if (num_genTriggerEventFlag_) {
    num_genTriggerEventFlag_.reset();
  }
  if (den_genTriggerEventFlag_) {
    den_genTriggerEventFlag_.reset();
  }

  delete hltPrescale_;
}

void BPHMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) {
  // Initialize the GenericTriggerEventFlag
  if (num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on()) {
    num_genTriggerEventFlag_->initRun(iRun, iSetup);
  }
  if (den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on()) {
    den_genTriggerEventFlag_->initRun(iRun, iSetup);
  }

  // check if every HLT path specified in numerator and denominator has a valid match in the HLT Menu
  hltPathsAreValid_ = (num_genTriggerEventFlag_ && den_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() &&
                       den_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->allHLTPathsAreValid() &&
                       den_genTriggerEventFlag_->allHLTPathsAreValid());

  // if valid HLT paths are required,
  // create DQM outputs only if all paths are valid
  if (requireValidHLTPaths_ and (not hltPathsAreValid_)) {
    return;
  }

  std::string histname, histtitle, istnp, trMuPh;

  bool Ph_ = false;
  if (enum_ == 7)
    Ph_ = true;
  if (tnp_)
    istnp = "Tag_and_Probe/";
  else
    istnp = "";

  std::string currentFolder = folderName_ + istnp;
  ibooker.setCurrentFolder(currentFolder);

  if (trOrMu_)
    trMuPh = "tr";
  else if (Ph_)
    trMuPh = "ph";
  else
    trMuPh = "mu";

  if (enum_ == 7 || enum_ == 1 || enum_ == 9 || enum_ == 10) {
    histname = trMuPh + "Pt";
    histtitle = trMuPh + "_P_{t}";
    bookME(ibooker, muPt_, histname, histtitle, pt_variable_binning_);
    setMETitle(muPt_, trMuPh + "_Pt[GeV]", "events / 1 GeV");

    histname = trMuPh + "Phi";
    histtitle = trMuPh + "Phi";
    bookME(ibooker, muPhi_, histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
    setMETitle(muPhi_, trMuPh + "_#phi", "events / 0.1 rad");

    histname = trMuPh + "Eta";
    histtitle = trMuPh + "_Eta";
    bookME(ibooker, muEta_, histname, histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax);
    setMETitle(muEta_, trMuPh + "_#eta", "events / 0.2");

    if (enum_ == 9) {
      histname = "BMass";
      histtitle = "BMass";
      bookME(ibooker, BMass_, histname, histtitle, Bmass_binning_.nbins, mass_binning_.xmin, mass_binning_.xmax);
      setMETitle(BMass_, "B_#mass", "events /");
    }
  } else {
    if (enum_ != 8) {
      histname = trMuPh + "1Pt";
      histtitle = trMuPh + "1_P_{t}";
      bookME(ibooker, mu1Pt_, histname, histtitle, pt_variable_binning_);
      setMETitle(mu1Pt_, trMuPh + "_Pt[GeV]", "events / 1 GeV");

      histname = trMuPh + "1Phi";
      histtitle = trMuPh + "1Phi";
      bookME(ibooker, mu1Phi_, histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
      setMETitle(mu1Phi_, trMuPh + "_#phi", "events / 0.1 rad");

      histname = trMuPh + "1Eta";
      histtitle = trMuPh + "1_Eta";
      bookME(ibooker, mu1Eta_, histname, histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax);
      setMETitle(mu1Eta_, trMuPh + "_#eta", "events / 0.2");

      histname = trMuPh + "2Pt";
      histtitle = trMuPh + "2_P_{t}";
      bookME(ibooker, mu2Pt_, histname, histtitle, pt_variable_binning_);
      setMETitle(mu2Pt_, trMuPh + "_Pt[GeV]", "events / 1 GeV");

      histname = trMuPh + "2Phi";
      histtitle = trMuPh + "2Phi";
      bookME(ibooker, mu2Phi_, histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
      setMETitle(mu2Phi_, trMuPh + "_#phi", "events / 0.1 rad");

      histname = trMuPh + "2Eta";
      histtitle = trMuPh + "2_Eta";
      bookME(ibooker, mu2Eta_, histname, histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax);
      setMETitle(mu2Eta_, trMuPh + "_#eta", "events / 0.2");
      if (enum_ == 11) {
        histname = "BMass";
        histtitle = "BMass";
        bookME(ibooker, BMass_, histname, histtitle, Bmass_binning_.nbins, mass_binning_.xmin, mass_binning_.xmax);
        setMETitle(BMass_, "B_#mass", "events /");
      }
    }
    if (enum_ == 6) {
      histname = trMuPh + "3Eta";
      histtitle = trMuPh + "3Eta";
      bookME(ibooker, mu3Eta_, histname, histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax);
      setMETitle(mu3Eta_, trMuPh + "3#eta", "events / 0.2");

      histname = trMuPh + "3Pt";
      histtitle = trMuPh + "3_P_{t}";
      bookME(ibooker, mu3Pt_, histname, histtitle, pt_variable_binning_);
      setMETitle(mu3Pt_, trMuPh + "3_Pt[GeV]", "events / 1 GeV");

      histname = trMuPh + "3Phi";
      histtitle = trMuPh + "3Phi";
      bookME(ibooker, mu3Phi_, histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
      setMETitle(mu3Phi_, trMuPh + "3_#phi", "events / 0.1 rad");

    } else if (enum_ == 2 || enum_ == 4 || enum_ == 5 || enum_ == 8) {
      histname = "DiMuEta";
      histtitle = "DiMuEta";
      bookME(ibooker, DiMuEta_, histname, histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax);
      setMETitle(DiMuEta_, "DiMu#eta", "events / 0.2");

      histname = "DiMuPt";
      histtitle = "DiMu_P_{t}";
      bookME(ibooker, DiMuPt_, histname, histtitle, dMu_pt_variable_binning_);
      setMETitle(DiMuPt_, "DiMu_Pt[GeV]", "events / 1 GeV");

      histname = "DiMuPhi";
      histtitle = "DiMuPhi";
      bookME(ibooker, DiMuPhi_, histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
      setMETitle(DiMuPhi_, "DiMu_#phi", "events / 0.1 rad");

      if (enum_ == 4 || enum_ == 5) {
        histname = "DiMudR";
        histtitle = "DiMudR";
        bookME(ibooker, DiMudR_, histname, histtitle, dR_binning_.nbins, dR_binning_.xmin, dR_binning_.xmax);
        setMETitle(DiMudR_, "DiMu_#dR", "events /");

        if (enum_ == 4) {
          histname = "DiMuMass";
          histtitle = "DiMuMass";
          bookME(ibooker, DiMuMass_, histname, histtitle, mass_binning_.nbins, mass_binning_.xmin, mass_binning_.xmax);
          setMETitle(DiMuMass_, "DiMu_#mass", "events /");
        }
      } else if (enum_ == 8) {
        histname = "DiMuProb";
        histtitle = "DiMuProb";
        bookME(ibooker, DiMuProb_, histname, histtitle, prob_variable_binning_);
        setMETitle(DiMuProb_, "DiMu_#prob", "events /");

        histname = "DiMuPVcos";
        histtitle = "DiMuPVcos";
        bookME(ibooker, DiMuPVcos_, histname, histtitle, cos_binning_.nbins, cos_binning_.xmin, cos_binning_.xmax);
        setMETitle(DiMuPVcos_, "DiMu_#cosPV", "events /");

        histname = "DiMuDS";
        histtitle = "DiMuDS";
        bookME(ibooker, DiMuDS_, histname, histtitle, ds_binning_.nbins, ds_binning_.xmin, ds_binning_.xmax);
        setMETitle(DiMuDS_, "DiMu_#ds", "events /");

        histname = "DiMuDCA";
        histtitle = "DiMuDCA";
        bookME(ibooker, DiMuDCA_, histname, histtitle, dca_binning_.nbins, dca_binning_.xmin, dca_binning_.xmax);
        setMETitle(DiMuDCA_, "DiMu_#dca", "events /");
      }
    }  // if (enum_ == 2 || enum_ == 4 || enum_ == 5 || enum_ == 8)
  }

  bool changed = true;

  hltPrescale_->init(iRun, iSetup, "HLT", changed);
  hltConfig_ = hltPrescale_->hltConfigProvider();
}

void BPHMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  // if valid HLT paths are required,
  // analyze event only if all paths are valid
  if (requireValidHLTPaths_ and (not hltPathsAreValid_)) {
    return;
  }

  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByToken(bsToken_, beamSpot);
  if (!beamSpot.isValid()) {
    return;
  }

  edm::Handle<reco::MuonCollection> muoHandle;
  iEvent.getByToken(muoToken_, muoHandle);
  if (!muoHandle.isValid()) {
    return;
  }

  edm::Handle<reco::TrackCollection> trHandle;
  iEvent.getByToken(trToken_, trHandle);
  if (!trHandle.isValid()) {
    return;
  }

  edm::Handle<reco::PhotonCollection> phHandle;
  iEvent.getByToken(phToken_, phHandle);

  edm::Handle<edm::TriggerResults> handleTriggerTrigRes;

  const std::string& hltpath = getTriggerName(hltpaths_den[0]);
  const std::string& hltpath1 = getTriggerName(hltpaths_num[0]);

  double PrescaleWeight = 1.0;
  if (den_genTriggerEventFlag_->on() && den_genTriggerEventFlag_->accept(iEvent, iSetup) &&
      num_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->accept(iEvent, iSetup))
    PrescaleWeight = Prescale(hltpath1, hltpath, iEvent, iSetup, hltPrescale_);

  if (tnp_ > 0) {  //TnP method

    if (den_genTriggerEventFlag_->on() && !den_genTriggerEventFlag_->accept(iEvent, iSetup))
      return;
    iEvent.getByToken(hltInputTag_, handleTriggerEvent);
    if (handleTriggerEvent->sizeFilters() == 0)
      return;

    std::vector<reco::Muon> tagMuons;
    for (auto const& m : *muoHandle) {  // applying tag selection
      if (!matchToTrigger(hltpath, m))
        continue;
      if (muoSelection_ref(m))
        tagMuons.push_back(m);
    }

    for (int i = 0; i < int(tagMuons.size()); i++) {
      for (auto const& m : *muoHandle) {
        if (!matchToTrigger(hltpath, m))
          continue;
        if ((tagMuons[i].pt() == m.pt()))
          continue;  //not the same
        if ((tagMuons[i].p4() + m.p4()).M() > minmass_ &&
            (tagMuons[i].p4() + m.p4()).M() < maxmass_) {  //near to J/psi mass
          muPhi_.denominator->Fill(m.phi());
          muEta_.denominator->Fill(m.eta());
          muPt_.denominator->Fill(m.pt());
          if (muoSelection_(m) && num_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->accept(iEvent, iSetup)) {
            muPhi_.numerator->Fill(m.phi(), PrescaleWeight);
            muEta_.numerator->Fill(m.eta(), PrescaleWeight);
            muPt_.numerator->Fill(m.pt(), PrescaleWeight);
          }
        }
      }
    }

  } else {  // reference method

    if (den_genTriggerEventFlag_->on() && (!den_genTriggerEventFlag_->accept(iEvent, iSetup)))
      return;

    iEvent.getByToken(hltInputTag_, handleTriggerEvent);
    if (handleTriggerEvent->sizeFilters() == 0)
      return;

    for (auto const& m : *muoHandle) {
      if (!muoSelection_ref(m))
        continue;
      if (!matchToTrigger(hltpath, m))
        continue;

      for (auto const& m1 : *muoHandle) {
        if (!(m1.pt() > m.pt()))
          continue;
        if (ptCut_) {
          if (!muoSelection_(m1))
            continue;
        } else if (!muoSelection_ref(m1))
          continue;
        if (!matchToTrigger(hltpath, m1))
          continue;

        if (enum_ != 10) {
          if (!DMSelection_ref(m1.p4() + m.p4()))
            continue;
          if (m.charge() * m1.charge() > 0)
            continue;
        }

        // dimuon vertex reconstruction
        MagneticField const& magneticField = iSetup.getData(magneticFieldToken_);
        const reco::BeamSpot& vertexBeamSpot = *beamSpot;
        std::vector<reco::TransientTrack> j_tks;
        reco::TransientTrack mu1TT(m.track(), &magneticField);
        reco::TransientTrack mu2TT(m1.track(), &magneticField);
        j_tks.push_back(mu1TT);
        j_tks.push_back(mu2TT);
        KalmanVertexFitter jkvf;
        TransientVertex jtv = jkvf.vertex(j_tks);
        if (!jtv.isValid())
          continue;
        reco::Vertex jpsivertex = jtv;
        float dimuonCL = 0;
        if ((jpsivertex.chi2() >= 0) && (jpsivertex.ndof() > 0))
          dimuonCL = TMath::Prob(jpsivertex.chi2(), jpsivertex.ndof());
        math::XYZVector jpperp(m.px() + m1.px(), m.py() + m1.py(), 0.);
        GlobalPoint jVertex = jtv.position();
        GlobalError jerr = jtv.positionError();
        GlobalPoint displacementFromBeamspotJpsi(
            -1 * ((vertexBeamSpot.x0() - jVertex.x()) + (jVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dxdz()),
            -1 * ((vertexBeamSpot.y0() - jVertex.y()) + (jVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dydz()),
            0);
        reco::Vertex::Point vperpj(displacementFromBeamspotJpsi.x(), displacementFromBeamspotJpsi.y(), 0.);
        float jpsi_cos = vperpj.Dot(jpperp) / (vperpj.R() * jpperp.R());
        TrajectoryStateClosestToPoint mu1TS = mu1TT.impactPointTSCP();
        TrajectoryStateClosestToPoint mu2TS = mu2TT.impactPointTSCP();
        ClosestApproachInRPhi cApp;
        if (mu1TS.isValid() && mu2TS.isValid()) {
          if (!cApp.calculate(mu1TS.theState(), mu2TS.theState()))
            continue;
        } else
          continue;
        double DiMuMass = (m1.p4() + m.p4()).M();

        switch (
            enum_) {  // enum_ = 1...9, represents different sets of variables for different paths, we want to have different hists for different paths

          case 1:

            tnp_ = true;  // already filled hists for tnp method
            [[fallthrough]];
          case 2:

            if ((Jpsi_) && (!Upsilon_))
              if (DiMuMass > maxmassJpsi || DiMuMass < minmassJpsi)
                continue;
            if ((!Jpsi_) && (Upsilon_))
              if (DiMuMass > maxmassUpsilon || DiMuMass < minmassUpsilon)
                continue;
            if (dimuonCL < minprob)
              continue;

            mu1Phi_.denominator->Fill(m.phi());
            mu1Eta_.denominator->Fill(m.eta());
            mu1Pt_.denominator->Fill(m.pt());
            mu2Phi_.denominator->Fill(m1.phi());
            mu2Eta_.denominator->Fill(m1.eta());
            mu2Pt_.denominator->Fill(m1.pt());
            DiMuPt_.denominator->Fill((m1.p4() + m.p4()).Pt());
            DiMuEta_.denominator->Fill((m1.p4() + m.p4()).Eta());
            DiMuPhi_.denominator->Fill((m1.p4() + m.p4()).Phi());

            if (num_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->accept(iEvent, iSetup)) {
              if (!matchToTrigger(hltpath1, m1))
                continue;
              if (!matchToTrigger(hltpath1, m))
                continue;
              mu1Phi_.numerator->Fill(m.phi(), PrescaleWeight);
              mu1Eta_.numerator->Fill(m.eta(), PrescaleWeight);
              mu1Pt_.numerator->Fill(m.pt(), PrescaleWeight);
              mu2Phi_.numerator->Fill(m1.phi(), PrescaleWeight);
              mu2Eta_.numerator->Fill(m1.eta(), PrescaleWeight);
              mu2Pt_.numerator->Fill(m1.pt(), PrescaleWeight);
              DiMuPt_.numerator->Fill((m1.p4() + m.p4()).Pt(), PrescaleWeight);
              DiMuEta_.numerator->Fill((m1.p4() + m.p4()).Eta(), PrescaleWeight);
              DiMuPhi_.numerator->Fill((m1.p4() + m.p4()).Phi(), PrescaleWeight);
            }

            break;

          case 3:

            if ((Jpsi_) && (!Upsilon_))
              if (DiMuMass > maxmassJpsi || DiMuMass < minmassJpsi)
                continue;
            if ((!Jpsi_) && (Upsilon_))
              if (DiMuMass > maxmassUpsilon || DiMuMass < minmassUpsilon)
                continue;
            if (dimuonCL < minprob)
              continue;

            mu1Phi_.denominator->Fill(m.phi());
            mu1Eta_.denominator->Fill(m.eta());
            mu1Pt_.denominator->Fill(m.pt());
            mu2Phi_.denominator->Fill(m1.phi());
            mu2Eta_.denominator->Fill(m1.eta());
            mu2Pt_.denominator->Fill(m1.pt());

            if (num_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->accept(iEvent, iSetup)) {
              if (!matchToTrigger(hltpath1, m1))
                continue;
              if (!matchToTrigger(hltpath1, m))
                continue;
              mu1Phi_.numerator->Fill(m.phi(), PrescaleWeight);
              mu1Eta_.numerator->Fill(m.eta(), PrescaleWeight);
              mu1Pt_.numerator->Fill(m.pt(), PrescaleWeight);
              mu2Phi_.numerator->Fill(m1.phi(), PrescaleWeight);
              mu2Eta_.numerator->Fill(m1.eta(), PrescaleWeight);
              mu2Pt_.numerator->Fill(m1.pt(), PrescaleWeight);
            }

            break;

          case 4:

            if (dimuonCL < minprob)
              continue;

            // fill mass plots without selecting mass region
            DiMuMass_.denominator->Fill(DiMuMass);
            if (num_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->accept(iEvent, iSetup) &&
                !(seagull_ && m.charge() * deltaPhi(m.phi(), m1.phi()) > 0) && matchToTrigger(hltpath1, m1) &&
                matchToTrigger(hltpath1, m))
              DiMuMass_.numerator->Fill(DiMuMass, PrescaleWeight);

            if ((Jpsi_) && (!Upsilon_))
              if (DiMuMass > maxmassJpsi || DiMuMass < minmassJpsi)
                continue;
            if ((!Jpsi_) && (Upsilon_))
              if (DiMuMass > maxmassUpsilon || DiMuMass < minmassUpsilon)
                continue;

            mu1Phi_.denominator->Fill(m.phi());
            mu1Eta_.denominator->Fill(m.eta());
            mu1Pt_.denominator->Fill(m.pt());
            mu2Phi_.denominator->Fill(m1.phi());
            mu2Eta_.denominator->Fill(m1.eta());
            mu2Pt_.denominator->Fill(m1.pt());
            DiMuPt_.denominator->Fill((m1.p4() + m.p4()).Pt());
            DiMuEta_.denominator->Fill((m1.p4() + m.p4()).Eta());
            DiMuPhi_.denominator->Fill((m1.p4() + m.p4()).Phi());
            DiMudR_.denominator->Fill(reco::deltaR(m, m1));

            if (num_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->accept(iEvent, iSetup)) {
              if (seagull_ && m.charge() * deltaPhi(m.phi(), m1.phi()) > 0)
                continue;
              if (!matchToTrigger(hltpath1, m1))
                continue;
              if (!matchToTrigger(hltpath1, m))
                continue;
              mu1Phi_.numerator->Fill(m.phi(), PrescaleWeight);
              mu1Eta_.numerator->Fill(m.eta(), PrescaleWeight);
              mu1Pt_.numerator->Fill(m.pt(), PrescaleWeight);
              mu2Phi_.numerator->Fill(m1.phi(), PrescaleWeight);
              mu2Eta_.numerator->Fill(m1.eta(), PrescaleWeight);
              mu2Pt_.numerator->Fill(m1.pt(), PrescaleWeight);
              DiMuPt_.numerator->Fill((m1.p4() + m.p4()).Pt(), PrescaleWeight);
              DiMuEta_.numerator->Fill((m1.p4() + m.p4()).Eta(), PrescaleWeight);
              DiMuPhi_.numerator->Fill((m1.p4() + m.p4()).Phi(), PrescaleWeight);
              DiMudR_.numerator->Fill(reco::deltaR(m, m1), PrescaleWeight);
            }

            break;

          case 5:

            if (dimuonCL < minprob)
              continue;
            if ((Jpsi_) && (!Upsilon_))
              if (DiMuMass > maxmassJpsi || DiMuMass < minmassJpsi)
                continue;
            if ((!Jpsi_) && (Upsilon_))
              if (DiMuMass > maxmassUpsilon || DiMuMass < minmassUpsilon)
                continue;

            mu1Phi_.denominator->Fill(m.phi());
            mu1Eta_.denominator->Fill(m.eta());
            mu1Pt_.denominator->Fill(m.pt());
            mu2Phi_.denominator->Fill(m1.phi());
            mu2Eta_.denominator->Fill(m1.eta());
            mu2Pt_.denominator->Fill(m1.pt());
            DiMuPt_.denominator->Fill((m1.p4() + m.p4()).Pt());
            DiMuEta_.denominator->Fill((m1.p4() + m.p4()).Eta());
            DiMuPhi_.denominator->Fill((m1.p4() + m.p4()).Phi());
            DiMudR_.denominator->Fill(reco::deltaR(m, m1));

            if (num_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->accept(iEvent, iSetup)) {
              if (seagull_ && m.charge() * deltaPhi(m.phi(), m1.phi()) > 0)
                continue;
              if (!matchToTrigger(hltpath1, m1))
                continue;
              if (!matchToTrigger(hltpath1, m))
                continue;
              mu1Phi_.numerator->Fill(m.phi(), PrescaleWeight);
              mu1Eta_.numerator->Fill(m.eta(), PrescaleWeight);
              mu1Pt_.numerator->Fill(m.pt(), PrescaleWeight);
              mu2Phi_.numerator->Fill(m1.phi(), PrescaleWeight);
              mu2Eta_.numerator->Fill(m1.eta(), PrescaleWeight);
              mu2Pt_.numerator->Fill(m1.pt(), PrescaleWeight);
              DiMuPt_.numerator->Fill((m1.p4() + m.p4()).Pt(), PrescaleWeight);
              DiMuEta_.numerator->Fill((m1.p4() + m.p4()).Eta(), PrescaleWeight);
              DiMuPhi_.numerator->Fill((m1.p4() + m.p4()).Phi(), PrescaleWeight);
              DiMudR_.numerator->Fill(reco::deltaR(m, m1), PrescaleWeight);
            }

            break;

          case 6:

            if (dimuonCL < minprob)
              continue;
            if ((Jpsi_) && (!Upsilon_))
              if (DiMuMass > maxmassJpsi || DiMuMass < minmassJpsi)
                continue;
            if ((!Jpsi_) && (Upsilon_))
              if (DiMuMass > maxmassUpsilon || DiMuMass < minmassUpsilon)
                continue;

            for (auto const& m2 : *muoHandle) {
              if (m2.pt() == m.pt())
                continue;  // remove duplicates but do not introduce ordering
              if (m2.pt() == m1.pt())
                continue;  // -> m2 will be the non-resonant and non-vertexing muon in the triplet
              if (!matchToTrigger(hltpath, m2))
                continue;

              mu1Phi_.denominator->Fill(m.phi());
              mu1Eta_.denominator->Fill(m.eta());
              mu1Pt_.denominator->Fill(m.pt());
              mu2Phi_.denominator->Fill(m1.phi());
              mu2Eta_.denominator->Fill(m1.eta());
              mu2Pt_.denominator->Fill(m1.pt());
              mu3Phi_.denominator->Fill(m2.phi());
              mu3Eta_.denominator->Fill(m2.eta());
              mu3Pt_.denominator->Fill(m2.pt());

              if (num_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->accept(iEvent, iSetup)) {
                if (!matchToTrigger(hltpath1, m1))
                  continue;
                if (!matchToTrigger(hltpath1, m))
                  continue;
                if (!matchToTrigger(hltpath1, m2))
                  continue;
                mu1Phi_.numerator->Fill(m.phi(), PrescaleWeight);
                mu1Eta_.numerator->Fill(m.eta(), PrescaleWeight);
                mu1Pt_.numerator->Fill(m.pt(), PrescaleWeight);
                mu2Phi_.numerator->Fill(m1.phi(), PrescaleWeight);
                mu2Eta_.numerator->Fill(m1.eta(), PrescaleWeight);
                mu2Pt_.numerator->Fill(m1.pt(), PrescaleWeight);
                mu3Phi_.numerator->Fill(m2.phi(), PrescaleWeight);
                mu3Eta_.numerator->Fill(m2.eta(), PrescaleWeight);
                mu3Pt_.numerator->Fill(m2.pt(), PrescaleWeight);
              }
            }

            break;

          case 7:

            if (phHandle.isValid()) {
              if (!phHandle->empty())
                for (auto const& p : *phHandle) {
                  if (!matchToTrigger(hltpath, p))
                    continue;

                  phPhi_.denominator->Fill(p.phi());
                  phEta_.denominator->Fill(p.eta());
                  phPt_.denominator->Fill(p.pt());

                  if (num_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->accept(iEvent, iSetup)) {
                    if (!matchToTrigger(hltpath1, p))
                      continue;
                    if (!matchToTrigger(hltpath1, m))
                      continue;
                    if (!matchToTrigger(hltpath1, m1))
                      continue;
                    phPhi_.numerator->Fill(p.phi(), PrescaleWeight);
                    phEta_.numerator->Fill(p.eta(), PrescaleWeight);
                    phPt_.numerator->Fill(p.pt(), PrescaleWeight);
                  }
                }
            } else {
              // if Handle is not valid, because the InputTag has been mis-configured, then skip the event
              if (!phInputTag_.label().empty())
                return;
            }

            break;

          case 8:  //vtx monitoring, filling probability, DS, DCA, cos of pointing angle to the PV, eta, pT of dimuon

            if (displaced_)
              if ((displacementFromBeamspotJpsi.perp() / sqrt(jerr.rerr(displacementFromBeamspotJpsi))) < minDS)
                continue;
            if ((Jpsi_) && (!Upsilon_))
              if (DiMuMass > maxmassJpsi || DiMuMass < minmassJpsi)
                continue;
            if ((!Jpsi_) && (Upsilon_))
              if (DiMuMass > maxmassUpsilon || DiMuMass < minmassUpsilon)
                continue;

            // fill vtx-prob plots before selecting on this variable
            DiMuProb_.denominator->Fill(dimuonCL);
            if (num_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->accept(iEvent, iSetup) &&
                matchToTrigger(hltpath1, m1) && matchToTrigger(hltpath1, m))
              DiMuProb_.numerator->Fill(dimuonCL, PrescaleWeight);

            if (dimuonCL < minprob)
              continue;

            DiMuDS_.denominator->Fill(displacementFromBeamspotJpsi.perp() /
                                      sqrt(jerr.rerr(displacementFromBeamspotJpsi)));
            DiMuPVcos_.denominator->Fill(jpsi_cos);
            DiMuPt_.denominator->Fill((m1.p4() + m.p4()).Pt());
            DiMuEta_.denominator->Fill((m1.p4() + m.p4()).Eta());
            DiMuPhi_.denominator->Fill((m1.p4() + m.p4()).Phi());
            DiMuDCA_.denominator->Fill(cApp.distance());

            if (num_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->accept(iEvent, iSetup)) {
              if (!matchToTrigger(hltpath1, m1))
                continue;
              if (!matchToTrigger(hltpath1, m))
                continue;
              DiMuDS_.numerator->Fill(
                  displacementFromBeamspotJpsi.perp() / sqrt(jerr.rerr(displacementFromBeamspotJpsi)), PrescaleWeight);
              DiMuPVcos_.numerator->Fill(jpsi_cos, PrescaleWeight);
              DiMuPt_.numerator->Fill((m1.p4() + m.p4()).Pt(), PrescaleWeight);
              DiMuEta_.numerator->Fill((m1.p4() + m.p4()).Eta(), PrescaleWeight);
              DiMuPhi_.numerator->Fill((m1.p4() + m.p4()).Phi(), PrescaleWeight);
              DiMuDCA_.numerator->Fill(cApp.distance(), PrescaleWeight);
            }

            break;

          case 9:

            if (dimuonCL < minprob)
              continue;
            if (fabs(jpsi_cos) < mincos)
              continue;
            if ((displacementFromBeamspotJpsi.perp() / sqrt(jerr.rerr(displacementFromBeamspotJpsi))) < minDS)
              continue;

            if (trHandle.isValid())
              for (auto const& t : *trHandle) {
                if (!trSelection_ref(t))
                  continue;
                const reco::Track& itrk1 = t;
                if (reco::deltaR(t, m1) <= min_dR)
                  continue;
                if (reco::deltaR(t, m) <= min_dR)
                  continue;
                if (!itrk1.quality(reco::TrackBase::highPurity))
                  continue;

                // reconstruct B+ hadron
                reco::Particle::LorentzVector pB, p1, p2, p3;
                double trackMass2 = kaon_mass * kaon_mass;
                double MuMass2 = mu_mass * mu_mass;
                double e1 = sqrt(m.momentum().Mag2() + MuMass2);
                double e2 = sqrt(m1.momentum().Mag2() + MuMass2);
                double e3 = sqrt(itrk1.momentum().Mag2() + trackMass2);
                p1 = reco::Particle::LorentzVector(m.px(), m.py(), m.pz(), e1);
                p2 = reco::Particle::LorentzVector(m1.px(), m1.py(), m1.pz(), e2);
                p3 = reco::Particle::LorentzVector(itrk1.px(), itrk1.py(), itrk1.pz(), e3);
                pB = p1 + p2 + p3;
                if (pB.mass() > maxmassJpsiTk || pB.mass() < minmassJpsiTk)
                  continue;
                reco::TransientTrack trTT(itrk1, &magneticField);
                std::vector<reco::TransientTrack> t_tks;
                t_tks.push_back(mu1TT);
                t_tks.push_back(mu2TT);
                t_tks.push_back(trTT);
                KalmanVertexFitter kvf;
                TransientVertex tv = kvf.vertex(t_tks);
                reco::Vertex vertex = tv;
                if (!tv.isValid())
                  continue;
                float JpsiTkCL = 0;
                if ((vertex.chi2() >= 0.0) && (vertex.ndof() > 0))
                  JpsiTkCL = TMath::Prob(vertex.chi2(), vertex.ndof());
                math::XYZVector pperp(m.px() + m1.px() + itrk1.px(), m.py() + m1.py() + itrk1.py(), 0.);
                GlobalPoint secondaryVertex = tv.position();
                GlobalError err = tv.positionError();
                GlobalPoint displacementFromBeamspot(
                    -1 * ((vertexBeamSpot.x0() - secondaryVertex.x()) +
                          (secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dxdz()),
                    -1 * ((vertexBeamSpot.y0() - secondaryVertex.y()) +
                          (secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dydz()),
                    0);
                reco::Vertex::Point vperp(displacementFromBeamspot.x(), displacementFromBeamspot.y(), 0.);
                float jpsiKcos = vperp.Dot(pperp) / (vperp.R() * pperp.R());
                if (JpsiTkCL < minprob)
                  continue;
                if (fabs(jpsiKcos) < mincos)
                  continue;
                if ((displacementFromBeamspot.perp() / sqrt(err.rerr(displacementFromBeamspot))) < minDS)
                  continue;

                muPhi_.denominator->Fill(t.phi());
                muEta_.denominator->Fill(t.eta());
                muPt_.denominator->Fill(t.pt());
                BMass_.denominator->Fill(pB.mass());

                if (num_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->accept(iEvent, iSetup)) {
                  if (!matchToTrigger(hltpath1, m1))
                    continue;
                  if (!matchToTrigger(hltpath1, m))
                    continue;
                  if (!matchToTrigger(hltpath1, t))
                    continue;
                  muPhi_.numerator->Fill(t.phi(), PrescaleWeight);
                  muEta_.numerator->Fill(t.eta(), PrescaleWeight);
                  muPt_.numerator->Fill(t.pt(), PrescaleWeight);
                  BMass_.numerator->Fill(pB.mass(), PrescaleWeight);
                }
              }

            break;

          case 10:

            if (trHandle.isValid())
              for (auto const& t : *trHandle) {
                if (!trSelection_ref(t))
                  continue;
                const reco::Track& itrk1 = t;
                if (reco::deltaR(t, m1) <= min_dR)
                  continue;
                if (reco::deltaR(t, m) <= min_dR)
                  continue;
                if (!itrk1.quality(reco::TrackBase::highPurity))
                  continue;

                // reconstruct Mu+TkMu structure
                reco::Particle::LorentzVector pB, p2, p3;
                double trackMass2 = kaon_mass * kaon_mass;
                double MuMass2 = mu_mass * mu_mass;
                double e2 = sqrt(m1.momentum().Mag2() + MuMass2);
                double e3 = sqrt(itrk1.momentum().Mag2() + trackMass2);
                p2 = reco::Particle::LorentzVector(m1.px(), m1.py(), m1.pz(), e2);
                p3 = reco::Particle::LorentzVector(itrk1.px(), itrk1.py(), itrk1.pz(), e3);
                pB = p2 + p3;
                if (pB.mass() > maxmassJpsiTk || pB.mass() < minmassJpsiTk)
                  continue;
                reco::TransientTrack trTT(itrk1, &magneticField);
                std::vector<reco::TransientTrack> t_tks;
                t_tks.push_back(mu2TT);
                t_tks.push_back(trTT);
                KalmanVertexFitter kvf;
                TransientVertex tv = kvf.vertex(t_tks);
                reco::Vertex vertex = tv;
                if (!tv.isValid())
                  continue;
                float JpsiTkCL = 0;
                if ((vertex.chi2() >= 0.0) && (vertex.ndof() > 0))
                  JpsiTkCL = TMath::Prob(vertex.chi2(), vertex.ndof());
                math::XYZVector pperp(m1.px() + itrk1.px(), m1.py() + itrk1.py(), 0.);
                GlobalPoint secondaryVertex = tv.position();
                GlobalError err = tv.positionError();
                GlobalPoint displacementFromBeamspot(
                    -1 * ((vertexBeamSpot.x0() - secondaryVertex.x()) +
                          (secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dxdz()),
                    -1 * ((vertexBeamSpot.y0() - secondaryVertex.y()) +
                          (secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dydz()),
                    0);
                reco::Vertex::Point vperp(displacementFromBeamspot.x(), displacementFromBeamspot.y(), 0.);
                if (JpsiTkCL < minprob)
                  continue;

                muPhi_.denominator->Fill(m1.phi());
                muEta_.denominator->Fill(m1.eta());
                muPt_.denominator->Fill(m1.pt());

                if (num_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->accept(iEvent, iSetup)) {
                  if (!matchToTrigger(hltpath1, m1))
                    continue;
                  if (!matchToTrigger(hltpath1, m))
                    continue;
                  if (!matchToTrigger(hltpath1, t))
                    continue;
                  muPhi_.numerator->Fill(m1.phi(), PrescaleWeight);
                  muEta_.numerator->Fill(m1.eta(), PrescaleWeight);
                  muPt_.numerator->Fill(m1.pt(), PrescaleWeight);
                }
              }

            break;

          case 11:

            if (dimuonCL < minprob)
              continue;
            if (fabs(jpsi_cos) < mincos)
              continue;
            if (displacementFromBeamspotJpsi.perp() / sqrt(jerr.rerr(displacementFromBeamspotJpsi)) < minDS)
              continue;

            if (trHandle.isValid())
              for (auto const& t : *trHandle) {
                if (!trSelection_ref(t))
                  continue;
                if ((reco::deltaR(t, m) <= min_dR))
                  continue;
                if ((reco::deltaR(t, m1) <= min_dR))
                  continue;

                for (auto const& t1 : *trHandle) {
                  if (&t - &(*trHandle)[0] >= &t1 - &(*trHandle)[0])
                    continue;  // not enough, need the following DeltaR checks
                  if (!trSelection_ref(t1))
                    continue;
                  if ((reco::deltaR(t1, m) <= min_dR))
                    continue;
                  if ((reco::deltaR(t1, m1) <= min_dR))
                    continue;
                  if ((reco::deltaR(t, t1) <= min_dR))
                    continue;
                  const reco::Track& itrk1 = t;
                  const reco::Track& itrk2 = t1;
                  if (!itrk1.quality(reco::TrackBase::highPurity))
                    continue;
                  if (!itrk2.quality(reco::TrackBase::highPurity))
                    continue;

                  // reconstruct Bs candidate
                  reco::Particle::LorentzVector pB, pTkTk, p1, p2, p3, p4;
                  double trackMass2 = kaon_mass * kaon_mass;
                  double MuMass2 = mu_mass * mu_mass;
                  double e1 = sqrt(m.momentum().Mag2() + MuMass2);
                  double e2 = sqrt(m1.momentum().Mag2() + MuMass2);
                  double e3 = sqrt(itrk1.momentum().Mag2() + trackMass2);
                  double e4 = sqrt(itrk2.momentum().Mag2() + trackMass2);
                  p1 = reco::Particle::LorentzVector(m.px(), m.py(), m.pz(), e1);
                  p2 = reco::Particle::LorentzVector(m1.px(), m1.py(), m1.pz(), e2);
                  p3 = reco::Particle::LorentzVector(itrk1.px(), itrk1.py(), itrk1.pz(), e3);
                  p4 = reco::Particle::LorentzVector(itrk2.px(), itrk2.py(), itrk2.pz(), e4);
                  pTkTk = p3 + p4;
                  if (pTkTk.mass() > maxmassTkTk || pTkTk.mass() < minmassTkTk)
                    continue;
                  pB = p1 + p2 + p3 + p4;
                  if (pB.mass() > maxmassJpsiTk || pB.mass() < minmassJpsiTk)
                    continue;
                  reco::TransientTrack mu1TT(m.track(), &magneticField);
                  reco::TransientTrack mu2TT(m1.track(), &magneticField);
                  reco::TransientTrack trTT(itrk1, &magneticField);
                  reco::TransientTrack tr1TT(itrk2, &magneticField);
                  std::vector<reco::TransientTrack> t_tks;
                  t_tks.push_back(mu1TT);
                  t_tks.push_back(mu2TT);
                  t_tks.push_back(trTT);
                  t_tks.push_back(tr1TT);
                  KalmanVertexFitter kvf;
                  TransientVertex tv = kvf.vertex(t_tks);  // this will compare the tracks
                  reco::Vertex vertex = tv;
                  if (!tv.isValid())
                    continue;
                  float JpsiTkCL = 0;
                  if ((vertex.chi2() >= 0.0) && (vertex.ndof() > 0))
                    JpsiTkCL = TMath::Prob(vertex.chi2(), vertex.ndof());
                  math::XYZVector pperp(
                      m.px() + m1.px() + itrk1.px() + itrk2.px(), m.py() + m1.py() + itrk1.py() + itrk2.py(), 0.);
                  GlobalPoint secondaryVertex = tv.position();
                  GlobalError err = tv.positionError();
                  GlobalPoint displacementFromBeamspot(
                      -1 * ((vertexBeamSpot.x0() - secondaryVertex.x()) +
                            (secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dxdz()),
                      -1 * ((vertexBeamSpot.y0() - secondaryVertex.y()) +
                            (secondaryVertex.z() - vertexBeamSpot.z0()) * vertexBeamSpot.dydz()),
                      0);
                  reco::Vertex::Point vperp(displacementFromBeamspot.x(), displacementFromBeamspot.y(), 0.);
                  float jpsiKcos = vperp.Dot(pperp) / (vperp.R() * pperp.R());
                  if (JpsiTkCL < minprob)
                    continue;
                  if (fabs(jpsiKcos) < mincos)
                    continue;
                  if ((displacementFromBeamspot.perp() / sqrt(err.rerr(displacementFromBeamspot))) < minDS)
                    continue;

                  mu1Phi_.denominator->Fill(t.phi());
                  mu1Eta_.denominator->Fill(t.eta());
                  mu1Pt_.denominator->Fill(t.pt());
                  mu2Phi_.denominator->Fill(t1.phi());
                  mu2Eta_.denominator->Fill(t1.eta());
                  mu2Pt_.denominator->Fill(t1.pt());
                  BMass_.denominator->Fill(pB.mass());

                  if (num_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->accept(iEvent, iSetup)) {
                    if (!matchToTrigger(hltpath1, m))
                      continue;
                    if (!matchToTrigger(hltpath1, m1))
                      continue;
                    if (!matchToTrigger(hltpath1, t))
                      continue;
                    if (!matchToTrigger(hltpath1, t1))
                      continue;
                    mu1Phi_.numerator->Fill(t.phi(), PrescaleWeight);
                    mu1Eta_.numerator->Fill(t.eta(), PrescaleWeight);
                    mu1Pt_.numerator->Fill(t.pt(), PrescaleWeight);
                    mu2Phi_.numerator->Fill(t1.phi(), PrescaleWeight);
                    mu2Eta_.numerator->Fill(t1.eta(), PrescaleWeight);
                    mu2Pt_.numerator->Fill(t1.pt(), PrescaleWeight);
                    BMass_.numerator->Fill(pB.mass(), PrescaleWeight);
                  }

                }  // for (auto const & t1 : *trHandle)
              }    // for (auto const & t : *trHandle)

            break;
        }
      }
    }
  }
}

void BPHMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("FolderName", "HLT/BPH/");
  desc.add<bool>("requireValidHLTPaths", true);

  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("photons", edm::InputTag("photons"));
  desc.add<edm::InputTag>("offlinePVs", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("muons", edm::InputTag("muons"));
  desc.add<edm::InputTag>("hltTriggerSummaryAOD", edm::InputTag("hltTriggerSummaryAOD", "", "HLT"));
  desc.add<std::string>("muoSelection", "");
  desc.add<std::string>("muoSelection_ref",
                        "isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & "
                        "innerTrack.hitPattern.numberOfValidPixelHits> 0");
  desc.add<std::string>(
      "muoSelection_tag",
      "isGlobalMuon && isPFMuon && isTrackerMuon && abs(eta) < 2.4 && innerTrack.hitPattern.numberOfValidPixelHits > 0 "
      "&& innerTrack.hitPattern.trackerLayersWithMeasurement > 5 && globalTrack.hitPattern.numberOfValidMuonHits > 0 "
      "&& globalTrack.normalizedChi2 < 10");  // tight selection for tag muon
  desc.add<std::string>("muoSelection_probe",
                        "isPFMuon & isGlobalMuon  & innerTrack.hitPattern.trackerLayersWithMeasurement>5 & "
                        "innerTrack.hitPattern.numberOfValidPixelHits> 0");
  desc.add<std::string>("trSelection_ref", "");
  desc.add<std::string>("DMSelection_ref", "Pt>4 & abs(eta)");

  desc.add<int>("nmuons", 1);
  desc.add<bool>("tnp", false);
  desc.add<int>("L3", 0);
  desc.add<int>("ptCut", 0);
  desc.add<int>("displaced", 0);
  desc.add<int>("trOrMu", 0);  // if =0, track param monitoring
  desc.add<int>("Jpsi", 0);
  desc.add<int>("Upsilon", 0);
  desc.add<int>("enum", 1);  // 1...9, 9 sets of variables to be filled, depends on the hlt path
  desc.add<int>("seagull", 1);
  desc.add<double>("maxmass", 3.596);
  desc.add<double>("minmass", 2.596);
  desc.add<double>("maxmassJpsi", 3.2);
  desc.add<double>("minmassJpsi", 3.);
  desc.add<double>("maxmassUpsilon", 10.0);
  desc.add<double>("minmassUpsilon", 8.8);
  desc.add<double>("maxmassTkTk", 10);
  desc.add<double>("minmassTkTk", 0);
  desc.add<double>("maxmassJpsiTk", 5.46);
  desc.add<double>("minmassJpsiTk", 5.1);
  desc.add<double>("kaon_mass", 0.493677);
  desc.add<double>("mu_mass", 0.1056583745);
  desc.add<double>("min_dR", 0.001);
  desc.add<double>("max_dR", 1.4);
  desc.add<double>("minprob", 0.005);
  desc.add<double>("mincos", 0.95);
  desc.add<double>("minDS", 3.);
  desc.add<unsigned int>("stageL1Trigger", 1);

  edm::ParameterSetDescription genericTriggerEventPSet;
  genericTriggerEventPSet.add<bool>("andOr");
  genericTriggerEventPSet.add<edm::InputTag>("dcsInputTag", edm::InputTag("scalersRawToDigi"));
  genericTriggerEventPSet.add<edm::InputTag>("hltInputTag", edm::InputTag("TriggerResults::HLT"));
  genericTriggerEventPSet.add<std::vector<int>>("dcsPartitions", {});
  genericTriggerEventPSet.add<bool>("andOrDcs", false);
  genericTriggerEventPSet.add<bool>("errorReplyDcs", true);
  genericTriggerEventPSet.add<std::string>("dbLabel", "");
  genericTriggerEventPSet.add<bool>("andOrHlt", true);
  genericTriggerEventPSet.add<bool>("andOrL1", true);
  genericTriggerEventPSet.add<std::vector<std::string>>("hltPaths", {});
  genericTriggerEventPSet.add<std::vector<std::string>>("l1Algorithms", {});
  genericTriggerEventPSet.add<std::string>("hltDBKey", "");
  genericTriggerEventPSet.add<bool>("errorReplyHlt", false);
  genericTriggerEventPSet.add<bool>("errorReplyL1", true);
  genericTriggerEventPSet.add<bool>("l1BeforeMask", true);
  genericTriggerEventPSet.add<unsigned int>("verbosityLevel", 0);
  desc.add<edm::ParameterSetDescription>("numGenericTriggerEventPSet", genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("denGenericTriggerEventPSet", genericTriggerEventPSet);

  edm::ParameterSetDescription histoPSet;
  edm::ParameterSetDescription phiPSet;
  edm::ParameterSetDescription etaPSet;
  edm::ParameterSetDescription ptPSet;
  edm::ParameterSetDescription dMu_ptPSet;
  edm::ParameterSetDescription d0PSet;
  edm::ParameterSetDescription z0PSet;
  edm::ParameterSetDescription dRPSet;
  edm::ParameterSetDescription massPSet;
  edm::ParameterSetDescription BmassPSet;
  edm::ParameterSetDescription dcaPSet;
  edm::ParameterSetDescription dsPSet;
  edm::ParameterSetDescription cosPSet;
  edm::ParameterSetDescription probPSet;
  edm::ParameterSetDescription TCoPSet;
  edm::ParameterSetDescription PUPSet;
  fillHistoPSetDescription(phiPSet);
  fillHistoPSetDescription(ptPSet);
  fillHistoPSetDescription(dMu_ptPSet);
  fillHistoPSetDescription(etaPSet);
  fillHistoPSetDescription(z0PSet);
  fillHistoPSetDescription(d0PSet);
  fillHistoPSetDescription(dRPSet);
  fillHistoPSetDescription(massPSet);
  fillHistoPSetDescription(BmassPSet);
  fillHistoPSetDescription(dcaPSet);
  fillHistoPSetDescription(dsPSet);
  fillHistoPSetDescription(cosPSet);
  fillHistoPSetDescription(probPSet);
  histoPSet.add<std::vector<double>>("ptBinning", {-0.5, 0, 2, 4, 8, 10, 12, 16, 20, 25, 30, 35, 40, 50});
  histoPSet.add<std::vector<double>>("dMuPtBinning", {6, 8, 12, 16, 20, 25, 30, 35, 40, 50, 70});
  histoPSet.add<std::vector<double>>("probBinning",
                                     {0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0});
  histoPSet.add<edm::ParameterSetDescription>("d0PSet", d0PSet);
  histoPSet.add<edm::ParameterSetDescription>("etaPSet", etaPSet);
  histoPSet.add<edm::ParameterSetDescription>("phiPSet", phiPSet);
  histoPSet.add<edm::ParameterSetDescription>("z0PSet", z0PSet);
  histoPSet.add<edm::ParameterSetDescription>("dRPSet", dRPSet);
  histoPSet.add<edm::ParameterSetDescription>("massPSet", massPSet);
  histoPSet.add<edm::ParameterSetDescription>("BmassPSet", BmassPSet);
  histoPSet.add<edm::ParameterSetDescription>("dcaPSet", dcaPSet);
  histoPSet.add<edm::ParameterSetDescription>("dsPSet", dsPSet);
  histoPSet.add<edm::ParameterSetDescription>("cosPSet", cosPSet);
  desc.add<edm::ParameterSetDescription>("histoPSet", histoPSet);

  descriptions.add("bphMonitoring", desc);
}

std::string BPHMonitor::getTriggerName(std::string partialName) {
  const std::string trigger_name_tmp = partialName.substr(0, partialName.find("v*"));
  const unsigned int Ntriggers(hltConfig_.size());
  std::string trigger_name = "";
  for (unsigned int i = 0; i < Ntriggers; i++) {
    trigger_name = hltConfig_.triggerName(i);
    if (trigger_name.find(trigger_name_tmp) != std::string::npos)
      break;
  }

  return trigger_name;
}

template <typename T>
bool BPHMonitor::matchToTrigger(const std::string& theTriggerName, T t) {
  bool matched = false;
  //validity check
  if (!hltConfig_.inited())
    return false;

  //Find the precise trigger name
  std::string trigger_name = getTriggerName(theTriggerName);
  const unsigned int trigger_index = hltConfig_.triggerIndex(trigger_name);

  //loop over all the modules for this trigger
  //by default use the last one
  unsigned int Nmodules = hltConfig_.size(trigger_index);
  const vector<string>& moduleLabels(hltConfig_.moduleLabels(trigger_index));
  unsigned int fIdx = 0;
  for (unsigned int i = 0; i < Nmodules; i++) {
    const unsigned int tmp_fIdx =
        handleTriggerEvent->filterIndex(edm::InputTag(moduleLabels[i], "", hltInputTag_1.process()));
    if (tmp_fIdx < handleTriggerEvent->sizeFilters())  //index of not used filters are set to sizeFilters()
    {
      fIdx = tmp_fIdx;
    }  //good index
  }

  //loop over all the objects in the filter of choice
  const trigger::Keys& KEYS(handleTriggerEvent->filterKeys(fIdx));
  const trigger::size_type nK(KEYS.size());
  const trigger::TriggerObjectCollection& TOC(handleTriggerEvent->getObjects());
  for (trigger::size_type i = 0; i != nK; ++i) {
    const trigger::TriggerObject& TO(TOC[KEYS[i]]);
    //perform matching: deltaR and pt check
    if ((reco::deltaR(t.eta(), t.phi(), TO.eta(), TO.phi()) <= 0.2) && (TMath::Abs(t.pt() - TO.pt()) < 0.12)) {
      matched = true;
    }
  }

  return matched;
}

double BPHMonitor::Prescale(const std::string hltpath1,
                            const std::string hltpath,
                            edm::Event const& iEvent,
                            edm::EventSetup const& iSetup,
                            HLTPrescaleProvider* hltPrescale_) {
  int PrescaleHLT_num = 1;
  int PrescaleHLT_den = 1;
  double Prescale_num = 1;
  double L1P = 1, HLTP = 1;
  bool flag = true;
  std::vector<bool> theSame_den;
  std::vector<bool> theSame_num;
  //retrieving HLT prescale
  PrescaleHLT_den = (hltPrescale_->prescaleValuesInDetail(iEvent, iSetup, hltpath)).second;
  PrescaleHLT_num = (hltPrescale_->prescaleValuesInDetail(iEvent, iSetup, hltpath1)).second;
  if (PrescaleHLT_den > 0 && PrescaleHLT_num > 0)
    HLTP = PrescaleHLT_num / std::__gcd(PrescaleHLT_num, PrescaleHLT_den);

  //retrieving L1 prescale
  //Checking if we have the same l1 seeds in den and num
  //taking into account that they can be written in different order in num and den
  //and some of them can be also switched off

  //check if for each den l1 there is the same l1 seed in num
  if (!(hltPrescale_->prescaleValuesInDetail(iEvent, iSetup, hltpath)).first.empty()) {
    for (size_t iSeed = 0; iSeed < (hltPrescale_->prescaleValuesInDetail(iEvent, iSetup, hltpath)).first.size();
         ++iSeed) {
      std::string l1_den = (hltPrescale_->prescaleValuesInDetail(iEvent, iSetup, hltpath)).first.at(iSeed).first;
      int l1_denp = (hltPrescale_->prescaleValuesInDetail(iEvent, iSetup, hltpath)).first.at(iSeed).second;
      if (l1_denp < 1)
        continue;
      flag = false;
      for (size_t iSeed1 = 0; iSeed1 < (hltPrescale_->prescaleValuesInDetail(iEvent, iSetup, hltpath1)).first.size();
           ++iSeed1) {
        std::string l1_num = (hltPrescale_->prescaleValuesInDetail(iEvent, iSetup, hltpath1)).first.at(iSeed1).first;
        int l1_nump = (hltPrescale_->prescaleValuesInDetail(iEvent, iSetup, hltpath1)).first.at(iSeed1).second;
        if (l1_num == l1_den && l1_nump >= 1)  //the same seed
        {
          flag = true;
          break;
        }
      }
      theSame_den.push_back(flag);
    }
  }
  //check if for each num l1 there is the same l1 seed in den
  if (!(hltPrescale_->prescaleValuesInDetail(iEvent, iSetup, hltpath1)).first.empty()) {
    for (size_t iSeed = 0; iSeed < (hltPrescale_->prescaleValuesInDetail(iEvent, iSetup, hltpath1)).first.size();
         ++iSeed) {
      std::string l1_num = (hltPrescale_->prescaleValuesInDetail(iEvent, iSetup, hltpath1)).first.at(iSeed).first;
      int l1_nump = (hltPrescale_->prescaleValuesInDetail(iEvent, iSetup, hltpath1)).first.at(iSeed).second;
      if (l1_nump < 1)
        continue;
      flag = false;
      for (size_t iSeed1 = 0; iSeed1 < (hltPrescale_->prescaleValuesInDetail(iEvent, iSetup, hltpath)).first.size();
           ++iSeed1) {
        std::string l1_den = (hltPrescale_->prescaleValuesInDetail(iEvent, iSetup, hltpath)).first.at(iSeed1).first;
        int l1_denp = (hltPrescale_->prescaleValuesInDetail(iEvent, iSetup, hltpath)).first.at(iSeed1).second;
        if (l1_den == l1_num && l1_denp >= 1)  //the same seed
        {
          flag = true;
          break;
        }
      }
      theSame_num.push_back(flag);
    }
  }
  flag = true;

  if (theSame_num.size() == theSame_den.size()) {
    for (size_t i = 0; i < theSame_num.size(); ++i) {
      if ((!theSame_num.at(i)) || (!theSame_den.at(i))) {
        flag = false;
        break;
      }
    }
  }

  if (flag && (theSame_num.size() == theSame_den.size())) {
    L1P = 1;  //den and num have the same set of l1 seeds
  } else {
    if (!(hltPrescale_->prescaleValuesInDetail(iEvent, iSetup, hltpath1)).first.empty()) {
      Prescale_num = 1;
      for (size_t iSeed = 0; iSeed < (hltPrescale_->prescaleValuesInDetail(iEvent, iSetup, hltpath1)).first.size();
           ++iSeed) {
        int l1 = (hltPrescale_->prescaleValuesInDetail(iEvent, iSetup, hltpath1)).first.at(iSeed).second;
        if (l1 < 1)
          continue;
        if (l1 == 1) {
          Prescale_num = 1;
          break;
        } else
          Prescale_num *= 1 - (1.0 / (l1));
      }
      if (Prescale_num != 1)
        Prescale_num = 1.0 / (1 - Prescale_num);
    }
    L1P = Prescale_num;
  }

  return L1P * HLTP;
}

DEFINE_FWK_MODULE(BPHMonitor);
