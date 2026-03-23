// system includes
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

// user includes
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Scouting/interface/Run3ScoutingElectron.h"
#include "DataFormats/Scouting/interface/Run3ScoutingMuon.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "ScoutingDQMUtils.h"

// ---------------------------------------------------------------------------
// Anonymous helpers
// ---------------------------------------------------------------------------
namespace {
  bool checkScoutingID(const Run3ScoutingMuon& obj) { return scoutingDQMUtils::scoutingMuonID(obj); }
  bool checkScoutingID(const Run3ScoutingElectron& obj) { return scoutingDQMUtils::scoutingElectronID(obj); }
}  // namespace

// ---------------------------------------------------------------------------
// scouting helpers - work on plain objects + optional ValueMap
// ---------------------------------------------------------------------------
namespace scouting {

  // Muon: charge is a direct member
  inline int chargeMu(const Run3ScoutingMuon& mu) { return mu.charge(); }

  // Electron: charge lives in the track vector; use ValueMap for best-track index when available
  inline int chargeEl(const Run3ScoutingElectron& el,
                      const edm::ValueMap<int>* vmBestIdx,
                      size_t collectionIndex,
                      const edm::Handle<Run3ScoutingElectronCollection>& handle) {
    if (vmBestIdx) {
      edm::Ref<Run3ScoutingElectronCollection> ref(handle, collectionIndex);
      int idx = (*vmBestIdx)[ref];
      return el.trkcharge()[idx];
    }
    return el.trkcharge()[0];
  }

  template <typename T>
  math::PtEtaPhiMLorentzVector p4(const T& obj, double mass) {
    return math::PtEtaPhiMLorentzVector(obj.pt(), obj.eta(), obj.phi(), mass);
  }

}  // namespace scouting

// ---------------------------------------------------------------------------
// Monitor class
// ---------------------------------------------------------------------------
class ScoutingDileptonMonitor : public DQMEDAnalyzer {
public:
  explicit ScoutingDileptonMonitor(const edm::ParameterSet&);
  ~ScoutingDileptonMonitor() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;

private:
  // ---- per-collection histogram set -------------------------------------
  struct MassHistos {
    MonitorElement* full{nullptr};
    MonitorElement* zwin{nullptr};
    MonitorElement* jpsiwin{nullptr};
    MonitorElement* barrel{nullptr};
    MonitorElement* endcap{nullptr};
  };

  // ---- helpers ----------------------------------------------------------
  template <typename T>
  void analyzeCollection(const edm::Event&,
                         const edm::EDGetTokenT<std::vector<T>>&,
                         const edm::EDGetTokenT<edm::ValueMap<int>>*,  // nullptr = no ValueMap (muons)
                         const StringCutObjectSelector<T>&,
                         MassHistos&,
                         bool doEtaSplit,
                         bool useID);

  void fillPairs(const std::vector<size_t>& indices,
                 const edm::Handle<std::vector<Run3ScoutingMuon>>& handle,
                 const edm::ValueMap<int>* /*unused*/,
                 MassHistos&,
                 bool doEtaSplit);

  void fillPairs(const std::vector<size_t>& indices,
                 const edm::Handle<std::vector<Run3ScoutingElectron>>& handle,
                 const edm::ValueMap<int>* vmBestIdx,
                 MassHistos&,
                 bool doEtaSplit);

  // ---- configuration ----------------------------------------------------
  const std::string outputInternalPath_;
  const double massMin_, massMax_;
  const int massBins_;
  const double zMin_, zMax_;
  const double jpsiMin_, jpsiMax_;
  const double barrelEta_;

  // ---- muons (Vtx) ------------------------------------------------------
  const bool doMuons_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingMuon>> muonToken_;
  const StringCutObjectSelector<Run3ScoutingMuon> muonCut_;
  MassHistos muonHistos_;
  const bool muonID_;

  // ---- muons (No Vtx) ---------------------------------------------------
  const bool doMuonsNoVtx_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingMuon>> muonNoVtxToken_;
  MassHistos muonNoVtxHistos_;

  // ---- electrons --------------------------------------------------------
  const bool doElectrons_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingElectron>> electronToken_;
  const edm::EDGetTokenT<edm::ValueMap<int>> vmBestTrackIndexToken_;
  const StringCutObjectSelector<Run3ScoutingElectron> electronCut_;
  MassHistos electronHistos_;
  const bool electronID_;
};

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
ScoutingDileptonMonitor::ScoutingDileptonMonitor(const edm::ParameterSet& iConfig)
    : outputInternalPath_{iConfig.getParameter<std::string>("OutputInternalPath")},
      massMin_(iConfig.getParameter<double>("massMin")),
      massMax_(iConfig.getParameter<double>("massMax")),
      massBins_(iConfig.getParameter<int>("massBins")),
      zMin_(iConfig.getParameter<double>("zMassMin")),
      zMax_(iConfig.getParameter<double>("zMassMax")),
      jpsiMin_(iConfig.getParameter<double>("jpsiMassMin")),
      jpsiMax_(iConfig.getParameter<double>("jpsiMassMax")),
      barrelEta_(iConfig.getParameter<double>("barrelEta")),
      doMuons_(iConfig.getParameter<bool>("doMuons")),
      muonToken_(consumes<std::vector<Run3ScoutingMuon>>(iConfig.getParameter<edm::InputTag>("muons"))),
      muonCut_(iConfig.getParameter<std::string>("muonCut")),
      muonID_(iConfig.getParameter<bool>("muonID")),
      doMuonsNoVtx_(iConfig.getParameter<bool>("doMuonsNoVtx")),
      muonNoVtxToken_(consumes<std::vector<Run3ScoutingMuon>>(iConfig.getParameter<edm::InputTag>("muonsNoVtx"))),
      doElectrons_(iConfig.getParameter<bool>("doElectrons")),
      electronToken_(consumes<std::vector<Run3ScoutingElectron>>(iConfig.getParameter<edm::InputTag>("electrons"))),
      vmBestTrackIndexToken_(consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("vmBestTrackIndex"))),
      electronCut_(iConfig.getParameter<std::string>("electronCut")),
      electronID_(iConfig.getParameter<bool>("electronID")) {}

// ---------------------------------------------------------------------------
// bookHistograms
// ---------------------------------------------------------------------------
void ScoutingDileptonMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  ibooker.setCurrentFolder(outputInternalPath_);

  auto bookSet = [&](const std::string& name, MassHistos& h, bool splitEta) {
    h.full = ibooker.book1D(
        name + "_mass", name + " opposite-charge invariant mass;M [GeV];Events", massBins_, massMin_, massMax_);
    h.zwin = ibooker.book1D(name + "_zMass", name + " Z window;M [GeV];Events", massBins_, zMin_, zMax_);
    h.jpsiwin =
        ibooker.book1D(name + "_jpsiMass", name + " J/Psi window;M [GeV];Events", massBins_, jpsiMin_, jpsiMax_);
    if (splitEta) {
      h.barrel = ibooker.book1D(name + "_barrelMass", name + " barrel;M [GeV];Events", massBins_, massMin_, massMax_);
      h.endcap = ibooker.book1D(name + "_endcapMass", name + " endcap;M [GeV];Events", massBins_, massMin_, massMax_);
    }
  };

  if (doMuons_)
    bookSet("muons", muonHistos_, true);
  if (doMuonsNoVtx_)
    bookSet("muonsNoVtx", muonNoVtxHistos_, true);
  if (doElectrons_)
    bookSet("electrons", electronHistos_, true);
}

// ---------------------------------------------------------------------------
// analyze
// ---------------------------------------------------------------------------
void ScoutingDileptonMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const&) {
  if (doMuons_) {
    edm::LogInfo("ScoutingDileptonMonitor") << "doing muons";
    analyzeCollection(iEvent, muonToken_, nullptr, muonCut_, muonHistos_, true, muonID_);
  }
  if (doMuonsNoVtx_) {
    edm::LogInfo("ScoutingDileptonMonitor") << "doing muons NoVtx";
    analyzeCollection(iEvent, muonNoVtxToken_, nullptr, muonCut_, muonNoVtxHistos_, true, muonID_);
  }
  if (doElectrons_) {
    edm::LogInfo("ScoutingDileptonMonitor") << "doing electrons";
    analyzeCollection(
        iEvent, electronToken_, &vmBestTrackIndexToken_, electronCut_, electronHistos_, true, electronID_);
  }
}

// ---------------------------------------------------------------------------
// analyzeCollection
// ---------------------------------------------------------------------------
template <typename T>
void ScoutingDileptonMonitor::analyzeCollection(const edm::Event& iEvent,
                                                const edm::EDGetTokenT<std::vector<T>>& token,
                                                const edm::EDGetTokenT<edm::ValueMap<int>>* bestTrackIndexToken,
                                                const StringCutObjectSelector<T>& cut,
                                                MassHistos& histos,
                                                bool doEtaSplit,
                                                bool useID) {
  edm::Handle<std::vector<T>> handle;
  iEvent.getByToken(token, handle);

  if (!handle.isValid()) {
    edm::LogWarning("ScoutingDileptonMonitor") << "Invalid Handle!";
    return;
  }

  const edm::ValueMap<int>* vmBestIdx = nullptr;
  if (bestTrackIndexToken != nullptr)
    vmBestIdx = &iEvent.get(*bestTrackIndexToken);

  edm::LogInfo("ScoutingDileptonMonitor") << "collection size: " << handle->size();

  std::vector<size_t> selected;
  selected.reserve(handle->size());
  for (size_t i = 0; i < handle->size(); ++i) {
    const auto& obj = (*handle)[i];
    if (!cut(obj))
      continue;
    if (useID && !checkScoutingID(obj))
      continue;
    selected.push_back(i);
  }

  fillPairs(selected, handle, vmBestIdx, histos, doEtaSplit);
}

// ---------------------------------------------------------------------------
// fillPairs - muon overload
// ---------------------------------------------------------------------------
void ScoutingDileptonMonitor::fillPairs(const std::vector<size_t>& indices,
                                        const edm::Handle<std::vector<Run3ScoutingMuon>>& handle,
                                        const edm::ValueMap<int>* /*unused*/,
                                        MassHistos& histos,
                                        bool doEtaSplit) {
  const size_t n = indices.size();
  edm::LogInfo("ScoutingDileptonMonitor") << "muon lepton size: " << n;

  for (size_t ii = 0; ii < n; ++ii) {
    for (size_t jj = ii + 1; jj < n; ++jj) {
      const auto& mu_i = (*handle)[indices[ii]];
      const auto& mu_j = (*handle)[indices[jj]];

      if (scouting::chargeMu(mu_i) * scouting::chargeMu(mu_j) >= 0)
        continue;

      const auto p4 = scouting::p4(mu_i, scoutingDQMUtils::MUON_MASS) + scouting::p4(mu_j, scoutingDQMUtils::MUON_MASS);
      const double mass = p4.mass();

      histos.full->Fill(mass);
      if (mass > zMin_ && mass < zMax_)
        histos.zwin->Fill(mass);
      if (mass > jpsiMin_ && mass < jpsiMax_)
        histos.jpsiwin->Fill(mass);
      if (doEtaSplit) {
        const bool barrel = std::abs(mu_i.eta()) < barrelEta_ && std::abs(mu_j.eta()) < barrelEta_;
        (barrel ? histos.barrel : histos.endcap)->Fill(mass);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// fillPairs - electron overload
// ---------------------------------------------------------------------------
void ScoutingDileptonMonitor::fillPairs(const std::vector<size_t>& indices,
                                        const edm::Handle<std::vector<Run3ScoutingElectron>>& handle,
                                        const edm::ValueMap<int>* vmBestIdx,
                                        MassHistos& histos,
                                        bool doEtaSplit) {
  const size_t n = indices.size();
  edm::LogInfo("ScoutingDileptonMonitor") << "electron lepton size: " << n;

  for (size_t ii = 0; ii < n; ++ii) {
    for (size_t jj = ii + 1; jj < n; ++jj) {
      const size_t idx_i = indices[ii];
      const size_t idx_j = indices[jj];
      const auto& el_i = (*handle)[idx_i];
      const auto& el_j = (*handle)[idx_j];

      const int q_i = scouting::chargeEl(el_i, vmBestIdx, idx_i, handle);
      const int q_j = scouting::chargeEl(el_j, vmBestIdx, idx_j, handle);
      if (q_i * q_j >= 0)
        continue;

      const auto p4 =
          scouting::p4(el_i, scoutingDQMUtils::ELECTRON_MASS) + scouting::p4(el_j, scoutingDQMUtils::ELECTRON_MASS);
      const double mass = p4.mass();

      histos.full->Fill(mass);
      if (mass > zMin_ && mass < zMax_)
        histos.zwin->Fill(mass);
      if (mass > jpsiMin_ && mass < jpsiMax_)
        histos.jpsiwin->Fill(mass);
      if (doEtaSplit) {
        const bool barrel = std::abs(el_i.eta()) < barrelEta_ && std::abs(el_j.eta()) < barrelEta_;
        (barrel ? histos.barrel : histos.endcap)->Fill(mass);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// fillDescriptions
// ---------------------------------------------------------------------------
void ScoutingDileptonMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("OutputInternalPath", "HLT/ScoutingOffline/DiLepton");
  desc.add<edm::InputTag>("muons", edm::InputTag("hltScoutingMuonPackerVtx"));
  desc.add<edm::InputTag>("muonsNoVtx", edm::InputTag("hltScoutingMuonPackerNoVtx"));
  desc.add<edm::InputTag>("electrons", edm::InputTag("hltScoutingEgammaPacker"));
  desc.add<edm::InputTag>("vmBestTrackIndex",
                          edm::InputTag("run3ScoutingElectronBestTrack", "Run3ScoutingElectronBestTrackIndex"));
  desc.add<bool>("doMuons", true);
  desc.add<bool>("doMuonsNoVtx", true);
  desc.add<bool>("doElectrons", true);
  desc.add<std::string>("muonCut", "");
  desc.add<std::string>("electronCut", "");
  desc.add<int>("massBins", 120);
  desc.add<double>("massMin", 0.0);
  desc.add<double>("massMax", 200.0);
  desc.add<double>("zMassMin", 70.0);
  desc.add<double>("zMassMax", 110.0);
  desc.add<double>("jpsiMassMin", 2.6);
  desc.add<double>("jpsiMassMax", 3.5);
  desc.add<double>("barrelEta", 1.479);
  desc.add<bool>("muonID", true);
  desc.add<bool>("electronID", true);
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(ScoutingDileptonMonitor);
