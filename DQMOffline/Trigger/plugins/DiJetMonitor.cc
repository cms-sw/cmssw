#include "DQMOffline/Trigger/plugins/DiJetMonitor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

// -----------------------------
//  constructors and destructor
// -----------------------------

DiJetMonitor::DiJetMonitor(const edm::ParameterSet& iConfig)
    : num_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"), consumesCollector(), *this)),
      den_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"), consumesCollector(), *this)) {
  folderName_ = iConfig.getParameter<std::string>("FolderName");
  dijetSrc_ = mayConsume<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("dijetSrc"));  //jet

  dijetpt_binning_ =
      getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("dijetPSet"));
  dijetptThr_binning_ = getHistoPSet(
      iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("dijetPtThrPSet"));

  ptcut_ = iConfig.getParameter<double>("ptcut");
}

void DiJetMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) {
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

  // Initialize the GenericTriggerEventFlag
  // Initialize the GenericTriggerEventFlag
  if (num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on())
    num_genTriggerEventFlag_->initRun(iRun, iSetup);
  if (den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on())
    den_genTriggerEventFlag_->initRun(iRun, iSetup);
}

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/Math/interface/deltaR.h"  // For Delta R
void DiJetMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  int Event = -999;
  Event = iEvent.id().event();
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
  jetptAvgbME_.denominator->Fill(pt_avg_b);

  if (dijet_selection(eta_1, phi_1, eta_2, phi_2, pt_1, pt_2, tag_id, probe_id, Event)) {
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
      jetAsyEtaME_.denominator->Fill(pt_asy, eta_1);
      jetEtaPhiME_.denominator->Fill(eta_1, phi_1);
    }

    if (num_genTriggerEventFlag_->on() && !num_genTriggerEventFlag_->accept(iEvent, iSetup))
      return;

    jetpt1ME_.numerator->Fill(pt_1);
    jetpt2ME_.numerator->Fill(pt_2);
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
      jetAsyEtaME_.numerator->Fill(pt_asy, eta_1);
      jetEtaPhiME_.numerator->Fill(eta_1, phi_1);
    }
  }
}

void DiJetMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("FolderName", "HLT/JME/Jets/AK4/PF");

  desc.add<edm::InputTag>("met", edm::InputTag("pfMet"));
  desc.add<edm::InputTag>("dijetSrc", edm::InputTag("ak4PFJets"));
  desc.add<edm::InputTag>("electrons", edm::InputTag("gedGsfElectrons"));
  desc.add<edm::InputTag>("muons", edm::InputTag("muons"));
  desc.add<int>("njets", 0);
  desc.add<int>("nelectrons", 0);
  desc.add<double>("ptcut", 20);

  edm::ParameterSetDescription genericTriggerEventPSet;
  genericTriggerEventPSet.add<bool>("andOr");
  genericTriggerEventPSet.add<edm::InputTag>("dcsInputTag", edm::InputTag("scalersRawToDigi"));
  genericTriggerEventPSet.add<std::vector<int> >("dcsPartitions", {});
  genericTriggerEventPSet.add<bool>("andOrDcs", false);
  genericTriggerEventPSet.add<bool>("errorReplyDcs", true);
  genericTriggerEventPSet.add<std::string>("dbLabel", "");
  genericTriggerEventPSet.add<bool>("andOrHlt", true);
  genericTriggerEventPSet.add<edm::InputTag>("hltInputTag", edm::InputTag("TriggerResults::HLT"));
  genericTriggerEventPSet.add<std::vector<std::string> >("hltPaths", {});
  genericTriggerEventPSet.add<bool>("errorReplyHlt", false);
  genericTriggerEventPSet.add<unsigned int>("verbosityLevel", 1);

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
  if (passeta && other_cuts)
    return true;
  else
    return false;
}

//------------------------------------------------------------------------//
// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DiJetMonitor);
