#include <string>
#include <vector>
#include <map>

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

class DiJetMonitor : public DQMEDAnalyzer, public TriggerDQMBase {
public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  DiJetMonitor(const edm::ParameterSet&);
  ~DiJetMonitor() throw() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  bool dijet_selection(double eta_1,
                       double phi_1,
                       double eta_2,
                       double phi_2,
                       double pt_1,
                       double pt_2,
                       int& tag_id,
                       int& probe_id,
                       int Event);

private:
  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  edm::EDGetTokenT<reco::PFMETCollection> metToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> eleToken_;
  edm::EDGetTokenT<reco::MuonCollection> muoToken_;
  edm::EDGetTokenT<reco::PFJetCollection> dijetSrc_;  // test for Jet

  MEbinning dijetpt_binning_;
  MEbinning dijetptThr_binning_;

  ObjME jetpt1ME_;
  ObjME jetpt2ME_;
  ObjME jetPhi1ME_;
  ObjME jetPhi2ME_;
  ObjME jetEta1ME_;
  ObjME jetEta2ME_;
  ObjME jetphiTagME_;
  ObjME jetptAvgaME_;
  ObjME jetptAvgaThrME_;
  ObjME jetptAvgbME_;
  ObjME jetptTagME_;
  ObjME jetptPrbME_;
  ObjME jetptAsyME_;
  ObjME jetetaPrbME_;
  ObjME jetetaTagME_;
  ObjME jetphiPrbME_;
  ObjME jetAsyEtaME_;
  ObjME jetEtaPhiME_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  int nmuons_;
  double ptcut_;

  // Define Phi Bin //
  const double DiJet_MAX_PHI = 3.2;
  // unsigned int DiJet_N_PHI = 64;
  unsigned int DiJet_N_PHI = 32;
  MEbinning dijet_phi_binning{DiJet_N_PHI, -DiJet_MAX_PHI, DiJet_MAX_PHI};
  // Define Eta Bin //
  const double DiJet_MAX_ETA = 5;
  //unsigned int DiJet_N_ETA = 50;
  unsigned int DiJet_N_ETA = 20;
  MEbinning dijet_eta_binning{DiJet_N_ETA, -DiJet_MAX_ETA, DiJet_MAX_ETA};

  const double MAX_asy = 1;
  const double MIN_asy = -1;
  //unsigned int N_asy = 100;
  unsigned int N_asy = 50;
  MEbinning asy_binning{N_asy, MIN_asy, MAX_asy};
};

DiJetMonitor::DiJetMonitor(const edm::ParameterSet& iConfig)
    : folderName_(iConfig.getParameter<std::string>("FolderName")),
      requireValidHLTPaths_(iConfig.getParameter<bool>("requireValidHLTPaths")),
      hltPathsAreValid_(false),
      dijetSrc_(mayConsume<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("dijetSrc"))),
      dijetpt_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("dijetPSet"))),
      dijetptThr_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("dijetPtThrPSet"))),
      num_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"), consumesCollector(), *this)),
      den_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"), consumesCollector(), *this)),
      ptcut_(iConfig.getParameter<double>("ptcut")) {}

DiJetMonitor::~DiJetMonitor() throw() {
  if (num_genTriggerEventFlag_) {
    num_genTriggerEventFlag_.reset();
  }
  if (den_genTriggerEventFlag_) {
    den_genTriggerEventFlag_.reset();
  }
}

void DiJetMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) {
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

  std::string histname, histtitle;
  std::string currentFolder = folderName_;
  ibooker.setCurrentFolder(currentFolder);

  histname = "jetpt1";
  histtitle = "leading Jet Pt";
  bookME(ibooker, jetpt1ME_, histname, histtitle, dijetpt_binning_.nbins, dijetpt_binning_.xmin, dijetpt_binning_.xmax);
  setMETitle(jetpt1ME_, "Pt_1 [GeV]", "events");

  histname = "jetpt2";
  histtitle = "second leading Jet Pt";
  bookME(ibooker, jetpt2ME_, histname, histtitle, dijetpt_binning_.nbins, dijetpt_binning_.xmin, dijetpt_binning_.xmax);
  setMETitle(jetpt2ME_, "Pt_2 [GeV]", "events");

  histname = "jetphi1";
  histtitle = "leading Jet Phi";
  bookME(
      ibooker, jetPhi1ME_, histname, histtitle, dijet_phi_binning.nbins, dijet_phi_binning.xmin, dijet_phi_binning.xmax);
  setMETitle(jetPhi1ME_, "Jet_Phi_1", "events");

  histname = "jetphi2";
  histtitle = "second leading Jet Phi";
  bookME(
      ibooker, jetPhi2ME_, histname, histtitle, dijet_phi_binning.nbins, dijet_phi_binning.xmin, dijet_phi_binning.xmax);
  setMETitle(jetPhi2ME_, "Jet_Phi_2", "events");

  histname = "jeteta1";
  histtitle = "leading Jet Eta";
  bookME(
      ibooker, jetEta1ME_, histname, histtitle, dijet_eta_binning.nbins, dijet_eta_binning.xmin, dijet_eta_binning.xmax);
  setMETitle(jetEta1ME_, "Jet_Eta_1", "events");

  histname = "jeteta2";
  histtitle = "second leading Jet Eta";
  bookME(
      ibooker, jetEta2ME_, histname, histtitle, dijet_eta_binning.nbins, dijet_eta_binning.xmin, dijet_eta_binning.xmax);
  setMETitle(jetEta2ME_, "Jet_Eta_2", "events");

  histname = "jetptAvgB";
  histtitle = "Pt average before offline selection";
  bookME(
      ibooker, jetptAvgbME_, histname, histtitle, dijetpt_binning_.nbins, dijetpt_binning_.xmin, dijetpt_binning_.xmax);
  setMETitle(jetptAvgbME_, "(pt_1 + pt_2)*0.5 [GeV]", "events");

  histname = "jetptAvgA";
  histtitle = "Pt average after offline selection";
  bookME(
      ibooker, jetptAvgaME_, histname, histtitle, dijetpt_binning_.nbins, dijetpt_binning_.xmin, dijetpt_binning_.xmax);
  setMETitle(jetptAvgaME_, "(pt_1 + pt_2)*0.5 [GeV]", "events");

  histname = "jetptAvgAThr";
  histtitle = "Pt average after offline selection";
  bookME(ibooker,
         jetptAvgaThrME_,
         histname,
         histtitle,
         dijetptThr_binning_.nbins,
         dijetptThr_binning_.xmin,
         dijetptThr_binning_.xmax);
  setMETitle(jetptAvgaThrME_, "(pt_1 + pt_2)*0.5 [GeV]", "events");

  histname = "jetptTag";
  histtitle = "Tag Jet Pt";
  bookME(
      ibooker, jetptTagME_, histname, histtitle, dijetpt_binning_.nbins, dijetpt_binning_.xmin, dijetpt_binning_.xmax);
  setMETitle(jetptTagME_, "Pt_tag [GeV]", "events ");

  histname = "jetptPrb";
  histtitle = "Probe Jet Pt";
  bookME(
      ibooker, jetptPrbME_, histname, histtitle, dijetpt_binning_.nbins, dijetpt_binning_.xmin, dijetpt_binning_.xmax);
  setMETitle(jetptPrbME_, "Pt_prb [GeV]", "events");

  histname = "jetptAsym";
  histtitle = "Jet Pt Asymetry";
  bookME(ibooker, jetptAsyME_, histname, histtitle, asy_binning.nbins, asy_binning.xmin, asy_binning.xmax);
  setMETitle(jetptAsyME_, "(pt_prb - pt_tag)/(pt_prb + pt_tag)", "events");

  histname = "jetetaPrb";
  histtitle = "Probe Jet eta";
  bookME(ibooker,
         jetetaPrbME_,
         histname,
         histtitle,
         dijet_eta_binning.nbins,
         dijet_eta_binning.xmin,
         dijet_eta_binning.xmax);
  setMETitle(jetetaPrbME_, "Eta_probe #eta", "events");

  histname = "jetetaTag";
  histtitle = "Tag Jet eta";
  bookME(ibooker,
         jetetaTagME_,
         histname,
         histtitle,
         dijet_eta_binning.nbins,
         dijet_eta_binning.xmin,
         dijet_eta_binning.xmax);
  setMETitle(jetetaTagME_, "Eta_tag #eta", "events");

  histname = "ptAsymVSetaPrb";
  histtitle = "Pt_Asym vs eta_prb";
  bookME(ibooker,
         jetAsyEtaME_,
         histname,
         histtitle,
         asy_binning.nbins,
         asy_binning.xmin,
         asy_binning.xmax,
         dijet_eta_binning.nbins,
         dijet_eta_binning.xmin,
         dijet_eta_binning.xmax);
  setMETitle(jetAsyEtaME_, "(pt_prb - pt_tag)/(pt_prb + pt_tag)", "Eta_probe #eta");

  histname = "etaPrbVSphiPrb";
  histtitle = "eta_prb vs phi_prb";
  bookME(ibooker,
         jetEtaPhiME_,
         histname,
         histtitle,
         dijet_eta_binning.nbins,
         dijet_eta_binning.xmin,
         dijet_eta_binning.xmax,
         dijet_phi_binning.nbins,
         dijet_phi_binning.xmin,
         dijet_phi_binning.xmax);
  setMETitle(jetEtaPhiME_, "Eta_probe #eta", "Phi_probe #phi");

  histname = "jetphiPrb";
  histtitle = "Probe Jet phi";
  bookME(ibooker,
         jetphiPrbME_,
         histname,
         histtitle,
         dijet_phi_binning.nbins,
         dijet_phi_binning.xmin,
         dijet_phi_binning.xmax);
  setMETitle(jetphiPrbME_, "Phi_probe #phi", "events");

  histname = "jetphiTag";
  histtitle = "Tag Jet phi";
  bookME(ibooker,
         jetphiTagME_,
         histname,
         histtitle,
         dijet_phi_binning.nbins,
         dijet_phi_binning.xmin,
         dijet_phi_binning.xmax);
  setMETitle(jetphiTagME_, "Phi_tag #phi", "events");
}

void DiJetMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  // if valid HLT paths are required,
  // analyze event only if all paths are valid
  if (requireValidHLTPaths_ and (not hltPathsAreValid_)) {
    return;
  }

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && !den_genTriggerEventFlag_->accept(iEvent, iSetup))
    return;

  std::vector<double> v_jetpt;
  std::vector<double> v_jeteta;
  std::vector<double> v_jetphi;

  v_jetpt.clear();
  v_jeteta.clear();
  v_jetphi.clear();

  edm::Handle<reco::PFJetCollection> offjets;
  iEvent.getByToken(dijetSrc_, offjets);
  if (!offjets.isValid()) {
    edm::LogWarning("DiJetMonitor") << "DiJet handle not valid \n";
    return;
  }
  for (reco::PFJetCollection::const_iterator ibegin = offjets->begin(), iend = offjets->end(), ijet = ibegin;
       ijet != iend;
       ++ijet) {
    if (ijet->pt() < ptcut_) {
      continue;
    }
    v_jetpt.push_back(ijet->pt());
    v_jeteta.push_back(ijet->eta());
    v_jetphi.push_back(ijet->phi());
  }
  if (v_jetpt.size() < 2) {
    return;
  }
  double pt_1 = v_jetpt[0];
  double eta_1 = v_jeteta[0];
  double phi_1 = v_jetphi[0];
  double pt_2 = v_jetpt[1];
  double eta_2 = v_jeteta[1];
  double phi_2 = v_jetphi[1];
  double pt_avg_b = (pt_1 + pt_2) * 0.5;
  int tag_id = -999, probe_id = -999;

  jetpt1ME_.denominator->Fill(pt_1);
  jetpt2ME_.denominator->Fill(pt_2);
  jetPhi1ME_.denominator->Fill(phi_1);
  jetPhi2ME_.denominator->Fill(phi_2);
  jetEta1ME_.denominator->Fill(eta_1);
  jetEta2ME_.denominator->Fill(eta_2);
  jetptAvgbME_.denominator->Fill(pt_avg_b);

  if (dijet_selection(eta_1, phi_1, eta_2, phi_2, pt_1, pt_2, tag_id, probe_id, iEvent.id().event())) {
    if (tag_id == 0 && probe_id == 1) {
      double pt_asy = (pt_2 - pt_1) / (pt_1 + pt_2);
      double pt_avg = (pt_1 + pt_2) * 0.5;
      jetptAvgaME_.denominator->Fill(pt_avg);
      jetptAvgaThrME_.denominator->Fill(pt_avg);
      jetptTagME_.denominator->Fill(pt_1);
      jetptPrbME_.denominator->Fill(pt_2);
      jetetaPrbME_.denominator->Fill(eta_2);
      jetetaTagME_.denominator->Fill(eta_1);
      jetptAsyME_.denominator->Fill(pt_asy);
      jetphiPrbME_.denominator->Fill(phi_2);
      jetphiTagME_.denominator->Fill(phi_1);
      jetAsyEtaME_.denominator->Fill(pt_asy, eta_2);
      jetEtaPhiME_.denominator->Fill(eta_2, phi_2);
    }
    if (tag_id == 1 && probe_id == 0) {
      double pt_asy = (pt_1 - pt_2) / (pt_2 + pt_1);
      double pt_avg = (pt_2 + pt_1) * 0.5;
      jetptAvgaME_.denominator->Fill(pt_avg);
      jetptAvgaThrME_.denominator->Fill(pt_avg);
      jetptTagME_.denominator->Fill(pt_2);
      jetptPrbME_.denominator->Fill(pt_1);
      jetetaPrbME_.denominator->Fill(eta_1);
      jetetaTagME_.denominator->Fill(eta_2);
      jetptAsyME_.denominator->Fill(pt_asy);
      jetphiPrbME_.denominator->Fill(phi_1);
      jetphiTagME_.denominator->Fill(phi_2);
      jetAsyEtaME_.denominator->Fill(pt_asy, eta_1);
      jetEtaPhiME_.denominator->Fill(eta_1, phi_1);
    }

    if (num_genTriggerEventFlag_->on() && !num_genTriggerEventFlag_->accept(iEvent, iSetup))
      return;

    jetpt1ME_.numerator->Fill(pt_1);
    jetpt2ME_.numerator->Fill(pt_2);
    jetPhi1ME_.numerator->Fill(phi_1);
    jetPhi2ME_.numerator->Fill(phi_2);
    jetEta1ME_.numerator->Fill(eta_1);
    jetEta2ME_.numerator->Fill(eta_2);
    jetptAvgbME_.numerator->Fill(pt_avg_b);

    if (tag_id == 0 && probe_id == 1) {
      double pt_asy = (pt_2 - pt_1) / (pt_1 + pt_2);
      double pt_avg = (pt_1 + pt_2) * 0.5;
      jetptAvgaME_.numerator->Fill(pt_avg);
      jetptAvgaThrME_.numerator->Fill(pt_avg);
      jetptTagME_.numerator->Fill(pt_1);
      jetptPrbME_.numerator->Fill(pt_2);
      jetetaPrbME_.numerator->Fill(eta_2);
      jetetaTagME_.numerator->Fill(eta_1);
      jetptAsyME_.numerator->Fill(pt_asy);
      jetphiPrbME_.numerator->Fill(phi_2);
      jetphiTagME_.numerator->Fill(phi_1);
      jetAsyEtaME_.numerator->Fill(pt_asy, eta_2);
      jetEtaPhiME_.numerator->Fill(eta_2, phi_2);
    }
    if (tag_id == 1 && probe_id == 0) {
      double pt_asy = (pt_1 - pt_2) / (pt_2 + pt_1);
      double pt_avg = (pt_2 + pt_1) * 0.5;
      jetptAvgaME_.numerator->Fill(pt_avg);
      jetptAvgaThrME_.numerator->Fill(pt_avg);
      jetptTagME_.numerator->Fill(pt_2);
      jetptPrbME_.numerator->Fill(pt_1);
      jetetaPrbME_.numerator->Fill(eta_1);
      jetetaTagME_.numerator->Fill(eta_2);
      jetptAsyME_.numerator->Fill(pt_asy);
      jetphiPrbME_.numerator->Fill(phi_1);
      jetphiTagME_.numerator->Fill(phi_2);
      jetAsyEtaME_.numerator->Fill(pt_asy, eta_1);
      jetEtaPhiME_.numerator->Fill(eta_1, phi_1);
    }
  }
}

//---- Additional DiJet offline selection------
bool DiJetMonitor::dijet_selection(double eta_1,
                                   double phi_1,
                                   double eta_2,
                                   double phi_2,
                                   double pt_1,
                                   double pt_2,
                                   int& tag_id,
                                   int& probe_id,
                                   int Event) {
  double etacut = 1.7;
  double phicut = 2.7;

  bool passeta = (std::abs(eta_1) < etacut || std::abs(eta_2) < etacut);  //check that one of the jets in the barrel

  float delta_phi_1_2 = (phi_1 - phi_2);
  bool other_cuts = (std::abs(delta_phi_1_2) >= phicut);  //check that jets are back to back

  if (std::abs(eta_1) < etacut && std::abs(eta_2) > etacut) {
    tag_id = 0;
    probe_id = 1;
  } else if (std::abs(eta_2) < etacut && std::abs(eta_1) > etacut) {
    tag_id = 1;
    probe_id = 0;
  } else if (std::abs(eta_2) < etacut && std::abs(eta_1) < etacut) {
    int numb = Event % 2;
    if (numb == 0) {
      tag_id = 0;
      probe_id = 1;
    }
    if (numb == 1) {
      tag_id = 1;
      probe_id = 0;
    }
  }

  return (passeta && other_cuts);
}

void DiJetMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("FolderName", "HLT/JME/Jets/AK4/PF");
  desc.add<bool>("requireValidHLTPaths", true);

  desc.add<edm::InputTag>("met", edm::InputTag("pfMet"));
  desc.add<edm::InputTag>("dijetSrc", edm::InputTag("ak4PFJets"));
  desc.add<edm::InputTag>("electrons", edm::InputTag("gedGsfElectrons"));
  desc.add<edm::InputTag>("muons", edm::InputTag("muons"));
  desc.add<int>("njets", 0);
  desc.add<int>("nelectrons", 0);
  desc.add<double>("ptcut", 20);

  edm::ParameterSetDescription genericTriggerEventPSet;
  GenericTriggerEventFlag::fillPSetDescription(genericTriggerEventPSet);

  desc.add<edm::ParameterSetDescription>("numGenericTriggerEventPSet", genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("denGenericTriggerEventPSet", genericTriggerEventPSet);

  edm::ParameterSetDescription histoPSet;
  edm::ParameterSetDescription dijetPSet;
  edm::ParameterSetDescription dijetPtThrPSet;
  fillHistoPSetDescription(dijetPSet);
  fillHistoPSetDescription(dijetPtThrPSet);
  histoPSet.add<edm::ParameterSetDescription>("dijetPSet", dijetPSet);
  histoPSet.add<edm::ParameterSetDescription>("dijetPtThrPSet", dijetPtThrPSet);
  std::vector<double> bins = {
      0.,   20.,  40.,  60.,  80.,  90.,  100., 110., 120., 130., 140., 150., 160.,
      170., 180., 190., 200., 220., 240., 260., 280., 300., 350., 400., 450., 1000.};  // DiJet pT Binning
  histoPSet.add<std::vector<double> >("jetptBinning", bins);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);
  desc.add<edm::ParameterSetDescription>("histoPSet", histoPSet);

  descriptions.add("dijetMonitoring", desc);
}

DEFINE_FWK_MODULE(DiJetMonitor);
