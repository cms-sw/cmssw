#include <string>
#include <vector>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

class PhotonMonitor : public DQMEDAnalyzer, public TriggerDQMBase {
public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  PhotonMonitor(const edm::ParameterSet&);
  ~PhotonMonitor() throw() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:
  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  edm::EDGetTokenT<reco::PFMETCollection> metToken_;
  edm::EDGetTokenT<reco::PFJetCollection> jetToken_;
  edm::EDGetTokenT<reco::GsfElectronCollection> eleToken_;
  edm::EDGetTokenT<reco::PhotonCollection> photonToken_;

  std::vector<double> photon_variable_binning_;
  std::vector<double> diphoton_mass_binning_;

  MEbinning photon_binning_;
  MEbinning ls_binning_;

  ObjME subphotonEtaME_;
  ObjME subphotonME_;
  ObjME subphotonPhiME_;
  ObjME subphotonME_variableBinning_;
  ObjME subphotonEtaPhiME_;
  ObjME subphotonr9ME_;
  ObjME subphotonHoverEME_;
  ObjME diphotonMassME_;

  ObjME photonEtaME_;
  ObjME photonME_;
  ObjME photonPhiME_;
  ObjME photonME_variableBinning_;
  ObjME photonVsLS_;
  ObjME photonEtaPhiME_;
  ObjME photonr9ME_;
  ObjME photonHoverEME_;

  double MAX_PHI1 = 3.2;
  unsigned int N_PHI1 = 64;
  const MEbinning phi_binning_1{N_PHI1, -MAX_PHI1, MAX_PHI1};

  double MAX_ETA = 1.4442;
  unsigned int N_ETA = 34;
  const MEbinning eta_binning_{N_ETA, -MAX_ETA, MAX_ETA};

  double MAX_r9 = 1;
  double MIN_r9 = 0;
  unsigned int N_r9 = 50;
  const MEbinning r9_binning_{N_r9, MIN_r9, MAX_r9};

  double MAX_hoe = 0.02;
  double MIN_hoe = 0;
  const MEbinning hoe_binning_{N_r9, MIN_hoe, MAX_hoe};

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::MET, true> metSelection_;
  StringCutObjectSelector<reco::PFJet, true> jetSelection_;
  StringCutObjectSelector<reco::GsfElectron, true> eleSelection_;
  StringCutObjectSelector<reco::Photon, true> photonSelection_;
  unsigned int njets_;
  unsigned int nphotons_;
  unsigned int nelectrons_;
};

PhotonMonitor::PhotonMonitor(const edm::ParameterSet& iConfig)
    : folderName_(iConfig.getParameter<std::string>("FolderName")),
      requireValidHLTPaths_(iConfig.getParameter<bool>("requireValidHLTPaths")),
      hltPathsAreValid_(false),
      metToken_(consumes<reco::PFMETCollection>(iConfig.getParameter<edm::InputTag>("met"))),
      jetToken_(mayConsume<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("jets"))),
      eleToken_(mayConsume<reco::GsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electrons"))),
      photonToken_(mayConsume<reco::PhotonCollection>(iConfig.getParameter<edm::InputTag>("photons"))),
      photon_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("photonBinning")),
      diphoton_mass_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("massBinning")),
      photon_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("photonPSet"))),
      ls_binning_(
          getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("lsPSet"))),
      num_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"), consumesCollector(), *this)),
      den_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"), consumesCollector(), *this)),
      metSelection_(iConfig.getParameter<std::string>("metSelection")),
      jetSelection_(iConfig.getParameter<std::string>("jetSelection")),
      eleSelection_(iConfig.getParameter<std::string>("eleSelection")),
      photonSelection_(iConfig.getParameter<std::string>("photonSelection")),
      njets_(iConfig.getParameter<unsigned int>("njets")),
      nphotons_(iConfig.getParameter<unsigned int>("nphotons")),
      nelectrons_(iConfig.getParameter<unsigned int>("nelectrons")) {}

PhotonMonitor::~PhotonMonitor() throw() {
  if (num_genTriggerEventFlag_) {
    num_genTriggerEventFlag_.reset();
  }
  if (den_genTriggerEventFlag_) {
    den_genTriggerEventFlag_.reset();
  }
}

void PhotonMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) {
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

  histname = "photon_pt";
  histtitle = "photon PT";
  bookME(ibooker, photonME_, histname, histtitle, photon_binning_.nbins, photon_binning_.xmin, photon_binning_.xmax);
  setMETitle(photonME_, "Photon pT [GeV]", "events / [GeV]");

  histname = "photon_pt_variable";
  histtitle = "photon PT";
  bookME(ibooker, photonME_variableBinning_, histname, histtitle, photon_variable_binning_);
  setMETitle(photonME_variableBinning_, "Photon pT [GeV]", "events / [GeV]");

  histname = "photonVsLS";
  histtitle = "photon pt vs LS";
  bookME(ibooker,
         photonVsLS_,
         histname,
         histtitle,
         ls_binning_.nbins,
         ls_binning_.xmin,
         ls_binning_.xmax,
         photon_binning_.xmin,
         photon_binning_.xmax);
  setMETitle(photonVsLS_, "LS", "Photon pT [GeV]");

  histname = "photon_phi";
  histtitle = "Photon phi";
  bookME(ibooker, photonPhiME_, histname, histtitle, phi_binning_1.nbins, phi_binning_1.xmin, phi_binning_1.xmax);
  setMETitle(photonPhiME_, "Photon #phi", "events / 0.1 rad");

  histname = "photon_eta";
  histtitle = "Photon eta";
  bookME(ibooker, photonEtaME_, histname, histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax);
  setMETitle(photonEtaME_, "Photon #eta", "events");

  histname = "photon_r9";
  histtitle = "Photon r9";
  bookME(ibooker, photonr9ME_, histname, histtitle, r9_binning_.nbins, r9_binning_.xmin, r9_binning_.xmax);
  setMETitle(photonr9ME_, "Photon r9", "events");

  histname = "photon_hoE";
  histtitle = "Photon hoverE";
  bookME(ibooker, photonHoverEME_, histname, histtitle, hoe_binning_.nbins, hoe_binning_.xmin, hoe_binning_.xmax);
  setMETitle(photonHoverEME_, "Photon hoE", "events");

  histname = "photon_etaphi";
  histtitle = "Photon eta-phi";
  bookME(ibooker,
         photonEtaPhiME_,
         histname,
         histtitle,
         eta_binning_.nbins,
         eta_binning_.xmin,
         eta_binning_.xmax,
         phi_binning_1.nbins,
         phi_binning_1.xmin,
         phi_binning_1.xmax);
  setMETitle(photonEtaPhiME_, "#eta", "#phi");

  // for diphotons
  if (nphotons_ > 1) {
    histname = "diphoton_mass";
    histtitle = "Diphoton mass";
    bookME(ibooker, diphotonMassME_, histname, histtitle, diphoton_mass_binning_);
    setMETitle(diphotonMassME_, "Diphoton mass", "events / 0.1");

    histname = "subphoton_pt";
    histtitle = "subphoton PT";
    bookME(
        ibooker, subphotonME_, histname, histtitle, photon_binning_.nbins, photon_binning_.xmin, photon_binning_.xmax);
    setMETitle(subphotonME_, "subPhoton pT [GeV]", "events / [GeV]");

    histname = "subphoton_eta";
    histtitle = "subPhoton eta";
    bookME(ibooker, subphotonEtaME_, histname, histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax);
    setMETitle(subphotonEtaME_, "subPhoton #eta", "events / 0.1");

    histname = "subphoton_phi";
    histtitle = "subPhoton phi";
    bookME(ibooker, subphotonPhiME_, histname, histtitle, phi_binning_1.nbins, phi_binning_1.xmin, phi_binning_1.xmax);
    setMETitle(subphotonPhiME_, "subPhoton #phi", "events / 0.1 rad");

    histname = "subphoton_r9";
    histtitle = "subPhoton r9";
    bookME(ibooker, subphotonr9ME_, histname, histtitle, r9_binning_.nbins, r9_binning_.xmin, r9_binning_.xmax);
    setMETitle(subphotonr9ME_, "subPhoton r9", "events");

    histname = "subphoton_hoE";
    histtitle = "subPhoton hoverE";
    bookME(ibooker, subphotonHoverEME_, histname, histtitle, hoe_binning_.nbins, hoe_binning_.xmin, hoe_binning_.xmax);
    setMETitle(subphotonHoverEME_, "subPhoton hoE", "events");

    histname = "subphoton_etaphi";
    histtitle = "subPhoton eta-phi";
    bookME(ibooker,
           subphotonEtaPhiME_,
           histname,
           histtitle,
           eta_binning_.nbins,
           eta_binning_.xmin,
           eta_binning_.xmax,
           phi_binning_1.nbins,
           phi_binning_1.xmin,
           phi_binning_1.xmax);
    setMETitle(subphotonEtaPhiME_, "#eta", "#phi");
  }
}

void PhotonMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  // if valid HLT paths are required,
  // analyze event only if all paths are valid
  if (requireValidHLTPaths_ and (not hltPathsAreValid_)) {
    return;
  }

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && !den_genTriggerEventFlag_->accept(iEvent, iSetup)) {
    return;
  }

  edm::Handle<reco::PFMETCollection> metHandle;
  iEvent.getByToken(metToken_, metHandle);
  reco::PFMET pfmet = metHandle->front();
  if (!metSelection_(pfmet))
    return;

  //float met = pfmet.pt();
  //  float phi = pfmet.phi();

  edm::Handle<reco::PFJetCollection> jetHandle;
  iEvent.getByToken(jetToken_, jetHandle);
  std::vector<reco::PFJet> jets;
  jets.clear();
  if (jetHandle->size() < njets_)
    return;
  for (auto const& j : *jetHandle) {
    if (jetSelection_(j))
      jets.push_back(j);
  }
  if (jets.size() < njets_)
    return;

  edm::Handle<reco::GsfElectronCollection> eleHandle;
  iEvent.getByToken(eleToken_, eleHandle);
  std::vector<reco::GsfElectron> electrons;
  if (eleHandle->size() < nelectrons_)
    return;
  for (auto const& e : *eleHandle) {
    if (eleSelection_(e))
      electrons.push_back(e);
  }
  if (electrons.size() < nelectrons_)
    return;

  edm::Handle<reco::PhotonCollection> photonHandle;
  iEvent.getByToken(photonToken_, photonHandle);
  std::vector<reco::Photon> photons;
  photons.clear();

  if (photonHandle->size() < nphotons_)
    return;
  for (auto const& p : *photonHandle) {
    if (photonSelection_(p))
      photons.push_back(p);
  }
  if (photons.size() < nphotons_)
    return;

  // filling histograms (denominator)
  int ls = iEvent.id().luminosityBlock();
  if (!(photons.empty()))

  {
    photonME_.denominator->Fill(photons[0].pt());
    photonME_variableBinning_.denominator->Fill(photons[0].pt());
    photonPhiME_.denominator->Fill(photons[0].phi());
    photonEtaME_.denominator->Fill(photons[0].eta());
    photonVsLS_.denominator->Fill(ls, photons[0].pt());
    photonEtaPhiME_.denominator->Fill(photons[0].eta(), photons[0].phi());
    photonr9ME_.denominator->Fill(photons[0].r9());
    photonHoverEME_.denominator->Fill(photons[0].hadTowOverEm());
  }

  if (nphotons_ > 1)
  //filling diphoton histograms
  {
    subphotonME_.denominator->Fill(photons[1].pt());
    subphotonEtaME_.denominator->Fill(photons[1].eta());
    subphotonPhiME_.denominator->Fill(photons[1].phi());
    subphotonEtaPhiME_.denominator->Fill(photons[1].eta(), photons[1].phi());
    subphotonr9ME_.denominator->Fill(photons[1].r9());
    subphotonHoverEME_.denominator->Fill(photons[1].hadTowOverEm());
    diphotonMassME_.denominator->Fill(
        sqrt(2 * photons[0].pt() * photons[1].pt() *
             (cosh(photons[0].eta() - photons[1].eta()) - cos(photons[0].phi() - photons[1].phi()))));
  }

  // applying selection for numerator
  if (num_genTriggerEventFlag_->on() && !num_genTriggerEventFlag_->accept(iEvent, iSetup))
    return;

  // filling histograms (num_genTriggerEventFlag_)
  if (!(photons.empty())) {
    photonME_.numerator->Fill(photons[0].pt());
    photonME_variableBinning_.numerator->Fill(photons[0].pt());
    photonPhiME_.numerator->Fill(photons[0].phi());
    photonEtaME_.numerator->Fill(photons[0].eta());
    photonVsLS_.numerator->Fill(ls, photons[0].pt());
    photonEtaPhiME_.numerator->Fill(photons[0].eta(), photons[0].phi());
    photonr9ME_.numerator->Fill(photons[0].r9());
    photonHoverEME_.numerator->Fill(photons[0].hadTowOverEm());
  }
  if (nphotons_ > 1)
  //filling diphoton histograms
  {
    subphotonME_.numerator->Fill(photons[1].pt());
    subphotonEtaME_.numerator->Fill(photons[1].eta());
    subphotonPhiME_.numerator->Fill(photons[1].phi());
    subphotonEtaPhiME_.numerator->Fill(photons[1].eta(), photons[1].phi());
    subphotonr9ME_.numerator->Fill(photons[1].r9());
    subphotonHoverEME_.numerator->Fill(photons[1].hadTowOverEm());
    diphotonMassME_.numerator->Fill(
        sqrt(2 * photons[0].pt() * photons[1].pt() *
             (cosh(photons[0].eta() - photons[1].eta()) - cos(photons[0].phi() - photons[1].phi()))));
  }
}

void PhotonMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("FolderName", "HLT/Photon");
  desc.add<bool>("requireValidHLTPaths", true);

  desc.add<edm::InputTag>("met", edm::InputTag("pfMet"));
  desc.add<edm::InputTag>("jets", edm::InputTag("ak4PFJetsCHS"));
  desc.add<edm::InputTag>("electrons", edm::InputTag("gedGsfElectrons"));
  desc.add<edm::InputTag>("photons", edm::InputTag("gedPhotons"));
  desc.add<std::string>("metSelection", "pt > 0");
  desc.add<std::string>("jetSelection", "pt > 0");
  desc.add<std::string>("eleSelection", "pt > 0");
  desc.add<std::string>(
      "photonSelection",
      "pt > 145 && eta<1.4442 && hadTowOverEm<0.0597 && full5x5_sigmaIetaIeta()<0.01031 && chargedHadronIso<1.295");
  //desc.add<std::string>("photonSelection", "pt > 145");
  desc.add<unsigned int>("njets", 0);
  desc.add<unsigned int>("nelectrons", 0);
  desc.add<unsigned int>("nphotons", 0);

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
  genericTriggerEventPSet.add<std::string>("hltDBKey", "");
  genericTriggerEventPSet.add<bool>("errorReplyHlt", false);
  genericTriggerEventPSet.add<unsigned int>("verbosityLevel", 1);

  desc.add<edm::ParameterSetDescription>("numGenericTriggerEventPSet", genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("denGenericTriggerEventPSet", genericTriggerEventPSet);

  edm::ParameterSetDescription histoPSet;
  edm::ParameterSetDescription metPSet;
  fillHistoPSetDescription(metPSet);
  histoPSet.add<edm::ParameterSetDescription>("photonPSet", metPSet);
  std::vector<double> bins = {0.,   20.,  40.,  60.,  80.,  90.,  100., 110., 120., 130., 140., 150., 160.,
                              170., 180., 190., 200., 220., 240., 260., 280., 300., 350., 400., 450., 1000.};
  histoPSet.add<std::vector<double> >("photonBinning", bins);
  std::vector<double> massbins = {90.,  91.,  92.,  93.,  94.,  95.,  96.,  97.,  98.,  99.,  100., 101., 102.,
                                  103., 104., 105., 106., 107., 108., 109., 110., 115., 120., 130., 150., 200.};
  histoPSet.add<std::vector<double> >("massBinning", massbins);
  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet", histoPSet);

  descriptions.add("photonMonitoring", desc);
}

DEFINE_FWK_MODULE(PhotonMonitor);
