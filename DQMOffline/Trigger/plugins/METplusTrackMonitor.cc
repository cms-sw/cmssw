#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

class METplusTrackMonitor : public DQMEDAnalyzer, public TriggerDQMBase {
public:
  typedef dqm::reco::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore DQMStore;

  METplusTrackMonitor(const edm::ParameterSet&);
  ~METplusTrackMonitor() noexcept(true) override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:
  bool getHLTObj(const edm::Handle<trigger::TriggerEvent>& trigSummary,
                 const edm::InputTag& filterTag,
                 trigger::TriggerObject& obj) const;

  const std::string folderName_;

  const bool requireValidHLTPaths_;
  bool hltPathsAreValid_;

  edm::EDGetTokenT<reco::CaloMETCollection> metToken_;
  edm::EDGetTokenT<reco::MuonCollection> muonToken_;
  edm::EDGetTokenT<reco::PFJetCollection> jetToken_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  edm::EDGetTokenT<trigger::TriggerEvent> theTrigSummary_;

  edm::InputTag hltMetTag_;
  edm::InputTag hltMetCleanTag_;
  edm::InputTag trackLegFilterTag_;

  std::vector<double> met_variable_binning_;
  std::vector<double> muonPt_variable_binning_;

  MEbinning met_binning_;
  MEbinning ls_binning_;
  MEbinning pt_binning_;
  MEbinning eta_binning_;
  MEbinning phi_binning_;

  ObjME metME_variableBinning_;
  ObjME metVsLS_;
  ObjME metPhiME_;
  ObjME deltaphimetj1ME_;
  ObjME metVsHltMet_;
  ObjME metVsHltMetClean_;

  ObjME muonPtME_variableBinning_;
  ObjME muonPtVsLS_;
  ObjME muonEtaME_;
  ObjME deltaphimetmuonME_;
  ObjME muonEtaVsPhi_;

  std::unique_ptr<GenericTriggerEventFlag> num_genTriggerEventFlag_;
  std::unique_ptr<GenericTriggerEventFlag> den_genTriggerEventFlag_;

  StringCutObjectSelector<reco::CaloMET, true> metSelection_;
  StringCutObjectSelector<reco::Muon, true> muonSelection_;
  StringCutObjectSelector<reco::PFJet, true> jetSelection_;
  StringCutObjectSelector<reco::Vertex, true> vtxSelection_;

  unsigned nmuons_;
  unsigned njets_;

  double leadJetEtaCut_;

  bool requireLeadMatched_;
  double maxMatchDeltaR_;
};

METplusTrackMonitor::METplusTrackMonitor(const edm::ParameterSet& iConfig)
    : folderName_(iConfig.getParameter<std::string>("FolderName")),
      requireValidHLTPaths_(iConfig.getParameter<bool>("requireValidHLTPaths")),
      hltPathsAreValid_(false),
      metToken_(consumes<reco::CaloMETCollection>(iConfig.getParameter<edm::InputTag>("met"))),
      muonToken_(consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
      jetToken_(consumes<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("jets"))),
      vtxToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      theTrigSummary_(consumes<trigger::TriggerEvent>(iConfig.getParameter<edm::InputTag>("trigSummary"))),
      hltMetTag_(iConfig.getParameter<edm::InputTag>("hltMetFilter")),
      hltMetCleanTag_(iConfig.getParameter<edm::InputTag>("hltMetCleanFilter")),
      trackLegFilterTag_(iConfig.getParameter<edm::InputTag>("trackLegFilter")),
      met_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("metBinning")),
      muonPt_variable_binning_(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("ptBinning")),
      met_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("metPSet"))),
      ls_binning_(
          getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("lsPSet"))),
      pt_binning_(
          getHistoPSet(iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("ptPSet"))),
      eta_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("etaPSet"))),
      phi_binning_(getHistoPSet(
          iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("phiPSet"))),
      num_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"), consumesCollector(), *this)),
      den_genTriggerEventFlag_(new GenericTriggerEventFlag(
          iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"), consumesCollector(), *this)),
      metSelection_(iConfig.getParameter<std::string>("metSelection")),
      muonSelection_(iConfig.getParameter<std::string>("muonSelection")),
      jetSelection_(iConfig.getParameter<std::string>("jetSelection")),
      vtxSelection_(iConfig.getParameter<std::string>("vtxSelection")),
      nmuons_(iConfig.getParameter<unsigned>("nmuons")),
      njets_(iConfig.getParameter<unsigned>("njets")),
      leadJetEtaCut_(iConfig.getParameter<double>("leadJetEtaCut")),
      requireLeadMatched_(iConfig.getParameter<bool>("requireLeadMatched")),
      maxMatchDeltaR_(iConfig.getParameter<double>("maxMatchDeltaR")) {}

void METplusTrackMonitor::bookHistograms(DQMStore::IBooker& ibooker,
                                         edm::Run const& iRun,
                                         edm::EventSetup const& iSetup) {
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

  // MET leg histograms
  histname = "met_variable";
  histtitle = "CaloMET";
  bookME(ibooker, metME_variableBinning_, histname, histtitle, met_variable_binning_);
  setMETitle(metME_variableBinning_, "CaloMET [GeV]", "events / [GeV]");

  histname = "metVsLS";
  histtitle = "CaloMET vs LS";
  bookME(ibooker,
         metVsLS_,
         histname,
         histtitle,
         ls_binning_.nbins,
         ls_binning_.xmin,
         ls_binning_.xmax,
         met_binning_.xmin,
         met_binning_.xmax);
  setMETitle(metVsLS_, "LS", "CaloMET [GeV]");

  histname = "metPhi";
  histtitle = "CaloMET phi";
  bookME(ibooker, metPhiME_, histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
  setMETitle(metPhiME_, "CaloMET #phi", "events / 0.2 rad");

  histname = "deltaphi_metjet1";
  histtitle = "dPhi(CaloMET, jet1)";
  bookME(ibooker, deltaphimetj1ME_, histname, histtitle, phi_binning_.nbins, 0, phi_binning_.xmax);
  setMETitle(deltaphimetj1ME_, "#Delta#phi (CaloMET, j1)", "events / 0.1 rad");

  histname = "metVsHltMet";
  histtitle = "CaloMET vs hltMet";
  bookME(ibooker,
         metVsHltMet_,
         histname,
         histtitle,
         met_binning_.nbins,
         met_binning_.xmin,
         met_binning_.xmax,
         met_binning_.nbins,
         met_binning_.xmin,
         met_binning_.xmax);
  setMETitle(metVsHltMet_, "hltMet (online) [GeV]", "CaloMET (offline) [GeV]");

  histname = "metVsHltMetClean";
  histtitle = "CaloMET vs hltMetClean";
  bookME(ibooker,
         metVsHltMetClean_,
         histname,
         histtitle,
         met_binning_.nbins,
         met_binning_.xmin,
         met_binning_.xmax,
         met_binning_.nbins,
         met_binning_.xmin,
         met_binning_.xmax);
  setMETitle(metVsHltMetClean_, "hltMetClean (online) [GeV]", "CaloMET (offline) [GeV]");

  // Track leg histograms

  histname = "muonPt_variable";
  histtitle = "Muon PT";
  bookME(ibooker, muonPtME_variableBinning_, histname, histtitle, muonPt_variable_binning_);
  setMETitle(muonPtME_variableBinning_, "Muon p_{T} [GeV]", "events / [GeV]");

  histname = "muonEta";
  histtitle = "Muon eta";
  bookME(ibooker, muonEtaME_, histname, histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax);
  setMETitle(muonEtaME_, "Muon #eta", "events / 0.2");

  histname = "deltaphi_muonmet";
  histtitle = "dPhi(Muon, CaloMET)";
  bookME(ibooker, deltaphimetmuonME_, histname, histtitle, phi_binning_.nbins, 0, phi_binning_.xmax);
  setMETitle(deltaphimetmuonME_, "#Delta#phi (Muon, CaloMET)", "events / 0.1 rad");

  histname = "muonEtaVsPhi";
  histtitle = "Muon eta vs phi";
  bookME(ibooker,
         muonEtaVsPhi_,
         histname,
         histtitle,
         phi_binning_.nbins,
         phi_binning_.xmin,
         phi_binning_.xmax,
         eta_binning_.nbins,
         eta_binning_.xmin,
         eta_binning_.xmax);
  setMETitle(muonEtaVsPhi_, "Muon #phi", "Muon #eta");

  histname = "muonPtVsLS";
  histtitle = "Muon PT vs LS";
  bookME(ibooker,
         muonPtVsLS_,
         histname,
         histtitle,
         ls_binning_.nbins,
         ls_binning_.xmin,
         ls_binning_.xmax,
         pt_binning_.xmin,
         pt_binning_.xmax);
  setMETitle(muonPtVsLS_, "LS", "Muon p_{T} [GeV]");
}

void METplusTrackMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  // if valid HLT paths are required,
  // analyze event only if all paths are valid
  if (requireValidHLTPaths_ and (not hltPathsAreValid_)) {
    return;
  }

  // Filter out events if Trigger Filtering is requested
  if (den_genTriggerEventFlag_->on() && !den_genTriggerEventFlag_->accept(iEvent, iSetup))
    return;

  edm::Handle<reco::CaloMETCollection> metHandle;
  iEvent.getByToken(metToken_, metHandle);
  reco::CaloMET caloMet = metHandle->front();
  if (!metSelection_(caloMet))
    return;

  float met = caloMet.pt();
  float metPhi = caloMet.phi();

  edm::Handle<reco::PFJetCollection> jetsHandle;
  iEvent.getByToken(jetToken_, jetsHandle);
  if (jetsHandle->size() < njets_)
    return;
  std::vector<reco::PFJet> jets;
  for (auto const& j : *jetsHandle) {
    if (jetSelection_(j))
      jets.push_back(j);
  }
  if (jets.size() < njets_)
    return;
  if (njets_ > 0 && !(jets.empty()) && fabs(jets[0].eta()) > leadJetEtaCut_)
    return;
  float deltaphi_metjet1 = !(jets.empty()) ? fabs(deltaPhi(caloMet.phi(), jets[0].phi())) : 10.0;

  edm::Handle<reco::VertexCollection> primaryVertices;
  iEvent.getByToken(vtxToken_, primaryVertices);
  if (primaryVertices->empty())
    return;
  const reco::Vertex* pv = nullptr;
  for (auto const& v : *primaryVertices) {
    if (!vtxSelection_(v))
      continue;
    pv = &v;
    break;
  }
  if (pv == nullptr)
    return;

  edm::Handle<reco::MuonCollection> muonHandle;
  iEvent.getByToken(muonToken_, muonHandle);
  if (muonHandle->size() < nmuons_)
    return;
  std::vector<reco::Muon> muons;
  for (auto const& m : *muonHandle) {
    bool passTightID =
        muon::isTightMuon(m, *pv) &&
        m.innerTrack()->hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::MISSING_INNER_HITS) == 0 &&
        m.innerTrack()->hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS) == 0;
    if (muonSelection_(m) && passTightID)
      muons.push_back(m);
  }
  if (muons.size() < nmuons_)
    return;

  // Filling MET leg histograms (denominator)
  metME_variableBinning_.denominator->Fill(met);
  metPhiME_.denominator->Fill(metPhi);
  deltaphimetj1ME_.denominator->Fill(deltaphi_metjet1);

  int ls = iEvent.id().luminosityBlock();
  metVsLS_.denominator->Fill(ls, met);

  // Apply the selection for the MET leg numerator
  edm::Handle<trigger::TriggerEvent> triggerSummary;
  iEvent.getByToken(theTrigSummary_, triggerSummary);
  if (!triggerSummary.isValid()) {
    edm::LogError("METplusTrackMonitor") << "Invalid collection: TriggerSummary"
                                         << "\n";
    return;
  }

  trigger::TriggerObject hltMet, hltMetClean;
  bool passesHltMetFilter = getHLTObj(triggerSummary, hltMetTag_, hltMet);
  bool passesHltMetCleanFilter = getHLTObj(triggerSummary, hltMetCleanTag_, hltMetClean);

  if (!passesHltMetFilter || !passesHltMetCleanFilter)
    return;

  // Filling MET leg histograms (numerator)
  metME_variableBinning_.numerator->Fill(met);
  metPhiME_.numerator->Fill(metPhi);
  deltaphimetj1ME_.numerator->Fill(deltaphi_metjet1);
  metVsLS_.numerator->Fill(ls, met);
  metVsHltMet_.numerator->Fill(hltMet.pt(), met);
  metVsHltMetClean_.numerator->Fill(hltMetClean.pt(), met);

  // Filling track leg histograms (denominator)
  double leadMuonPt = !(muons.empty()) ? muons[0].pt() : -1.0;
  double leadMuonEta = !(muons.empty()) ? muons[0].eta() : 10.0;
  double leadMuonPhi = !(muons.empty()) ? muons[0].phi() : 10.0;
  float deltaphi_metmuon = !(muons.empty()) ? fabs(deltaPhi(caloMet.phi(), muons[0].phi())) : 10.0;

  muonPtME_variableBinning_.denominator->Fill(leadMuonPt);
  muonPtVsLS_.denominator->Fill(ls, leadMuonPt);
  muonEtaME_.denominator->Fill(leadMuonEta);
  deltaphimetmuonME_.denominator->Fill(deltaphi_metmuon);
  muonEtaVsPhi_.denominator->Fill(leadMuonPhi, leadMuonEta);

  // Apply the selection for the track leg numerator
  trigger::TriggerObject isoTrk;
  bool passesTrackLegFilter = getHLTObj(triggerSummary, trackLegFilterTag_, isoTrk);

  // require track leg filter
  if (!passesTrackLegFilter)
    return;

  // if requested, require lead selected muon is matched to the track leg filter object
  if (requireLeadMatched_ && !(muons.empty()) && deltaR(muons[0], isoTrk) < maxMatchDeltaR_)
    return;

  // require the full HLT path is fired
  if (num_genTriggerEventFlag_->on() && !num_genTriggerEventFlag_->accept(iEvent, iSetup))
    return;

  // Filling track leg histograms (denominator)
  muonPtME_variableBinning_.numerator->Fill(leadMuonPt);
  muonPtVsLS_.numerator->Fill(ls, leadMuonPt);
  muonEtaME_.numerator->Fill(leadMuonEta);
  deltaphimetmuonME_.numerator->Fill(deltaphi_metmuon);
  muonEtaVsPhi_.numerator->Fill(leadMuonPhi, leadMuonEta);
}

bool METplusTrackMonitor::getHLTObj(const edm::Handle<trigger::TriggerEvent>& trigSummary,
                                    const edm::InputTag& filterTag,
                                    trigger::TriggerObject& obj) const {
  double leadingPt = -1.0;

  size_t filterIndex = trigSummary->filterIndex(filterTag);
  trigger::TriggerObjectCollection triggerObjects = trigSummary->getObjects();

  if (!(filterIndex >= trigSummary->sizeFilters())) {
    const trigger::Keys& keys = trigSummary->filterKeys(filterIndex);
    for (unsigned short key : keys) {
      trigger::TriggerObject foundObject = triggerObjects[key];
      if (foundObject.pt() > leadingPt) {
        obj = foundObject;
        leadingPt = obj.pt();
      }
    }
  }

  return (leadingPt > 0.0);
}

void METplusTrackMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("FolderName", "HLT/MET");
  desc.add<bool>("requireValidHLTPaths", true);

  desc.add<edm::InputTag>("met", edm::InputTag("caloMet"));
  desc.add<edm::InputTag>("jets", edm::InputTag("ak4PFJetsCHS"));
  desc.add<edm::InputTag>("electrons", edm::InputTag("gedGsfElectrons"));
  desc.add<edm::InputTag>("muons", edm::InputTag("muons"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("trigSummary", edm::InputTag("hltTriggerSummaryAOD"));
  desc.add<edm::InputTag>("hltMetFilter", edm::InputTag("hltMET105", "", "HLT"));
  desc.add<edm::InputTag>("hltMetCleanFilter", edm::InputTag("hltMETClean65", "", "HLT"));
  desc.add<edm::InputTag>("trackLegFilter", edm::InputTag("hltTrk50Filter", "", "HLT"));

  desc.add<std::string>("metSelection", "pt > 0");
  desc.add<std::string>("jetSelection", "pt > 0");
  desc.add<std::string>("muonSelection", "pt > 0");
  desc.add<std::string>("vtxSelection", "!isFake");
  desc.add<unsigned>("njets", 0);
  desc.add<unsigned>("nmuons", 0);
  desc.add<double>("leadJetEtaCut", 2.4);
  desc.add<bool>("requireLeadMatched", true);
  desc.add<double>("maxMatchDeltaR", 0.1);

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
  edm::ParameterSetDescription phiPSet;
  edm::ParameterSetDescription ptPSet;
  edm::ParameterSetDescription etaPSet;

  fillHistoPSetDescription(metPSet);
  fillHistoPSetDescription(phiPSet);
  fillHistoPSetDescription(ptPSet);
  fillHistoPSetDescription(etaPSet);

  histoPSet.add<edm::ParameterSetDescription>("metPSet", metPSet);
  histoPSet.add<edm::ParameterSetDescription>("phiPSet", phiPSet);
  histoPSet.add<edm::ParameterSetDescription>("ptPSet", ptPSet);
  histoPSet.add<edm::ParameterSetDescription>("etaPSet", etaPSet);

  std::vector<double> bins = {0.,   20.,  40.,  60.,  80.,  90.,  100., 110., 120., 130., 140., 150., 160.,
                              170., 180., 190., 200., 220., 240., 260., 280., 300., 350., 400., 450., 1000.};

  histoPSet.add<std::vector<double> >("metBinning", bins);
  histoPSet.add<std::vector<double> >("ptBinning", bins);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet", histoPSet);

  descriptions.add("metPlusTrackMonitoring", desc);
}

DEFINE_FWK_MODULE(METplusTrackMonitor);
