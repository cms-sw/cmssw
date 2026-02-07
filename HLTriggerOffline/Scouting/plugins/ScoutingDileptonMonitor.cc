// system includes
#include <type_traits>
#include <map>
#include <string>
#include <vector>

// user includes
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Scouting/interface/Run3ScoutingElectron.h"
#include "DataFormats/Scouting/interface/Run3ScoutingMuon.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "ScoutingDQMUtils.h"

namespace {
  bool checkScoutingID(const Run3ScoutingMuon& obj) { return scoutingDQMUtils::scoutingMuonID(obj); }

  bool checkScoutingID(const Run3ScoutingElectron& obj) { return scoutingDQMUtils::scoutingElectronID(obj); }
}  // namespace

namespace scouting {
  inline int charge(const Run3ScoutingMuon& mu) { return mu.charge(); }
  inline int charge(const Run3ScoutingElectron& el) { return el.trkcharge()[0]; }

  template <typename T>
  math::PtEtaPhiMLorentzVector p4(const T& obj, double mass) {
    return math::PtEtaPhiMLorentzVector(obj.pt(), obj.eta(), obj.phi(), mass);
  }
}  // namespace scouting

class ScoutingDileptonMonitor : public DQMEDAnalyzer {
public:
  explicit ScoutingDileptonMonitor(const edm::ParameterSet&);
  ~ScoutingDileptonMonitor() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;

private:
  // ---- internal structs -------------------------------------------------
  struct MassHistos {
    MonitorElement* full{nullptr};
    MonitorElement* zwin{nullptr};
    MonitorElement* barrel{nullptr};
    MonitorElement* endcap{nullptr};
  };

  // ---- helpers ----------------------------------------------------------
  template <typename T>
  void analyzeCollection(const edm::Event&,
                         const edm::EDGetTokenT<std::vector<T>>&,
                         const StringCutObjectSelector<T>&,
                         MassHistos&,
                         bool doEtaSplit,
                         bool useID);

  template <typename T>
  void fillPairs(const std::vector<const T*>&, MassHistos&, bool doEtaSplit);

  // ---- configuration ----------------------------------------------------
  const std::string outputInternalPath_;
  const double massMin_;
  const double massMax_;
  const int massBins_;
  const double zMin_;
  const double zMax_;
  const double barrelEta_;

  // ---- muons ------------------------------------------------------------
  const bool doMuons_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingMuon>> muonToken_;
  const StringCutObjectSelector<Run3ScoutingMuon> muonCut_;
  MassHistos muonHistos_;
  const bool muonID_;

  // ---- electrons --------------------------------------------------------
  const bool doElectrons_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingElectron>> electronToken_;
  const StringCutObjectSelector<Run3ScoutingElectron> electronCut_;
  MassHistos electronHistos_;
  const bool electronID_;
};

ScoutingDileptonMonitor::ScoutingDileptonMonitor(const edm::ParameterSet& iConfig)
    : outputInternalPath_{iConfig.getParameter<std::string>("OutputInternalPath")},
      massMin_(iConfig.getParameter<double>("massMin")),
      massMax_(iConfig.getParameter<double>("massMax")),
      massBins_(iConfig.getParameter<int>("massBins")),
      zMin_(iConfig.getParameter<double>("zMassMin")),
      zMax_(iConfig.getParameter<double>("zMassMax")),
      barrelEta_(iConfig.getParameter<double>("barrelEta")),

      doMuons_(iConfig.getParameter<bool>("doMuons")),
      muonToken_(consumes<std::vector<Run3ScoutingMuon>>(iConfig.getParameter<edm::InputTag>("muons"))),
      muonCut_(iConfig.getParameter<std::string>("muonCut")),
      muonID_(iConfig.getParameter<bool>("muonID")),

      doElectrons_(iConfig.getParameter<bool>("doElectrons")),
      electronToken_(consumes<std::vector<Run3ScoutingElectron>>(iConfig.getParameter<edm::InputTag>("electrons"))),
      electronCut_(iConfig.getParameter<std::string>("electronCut")),
      electronID_(iConfig.getParameter<bool>("electronID")) {}

void ScoutingDileptonMonitor::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  ibooker.setCurrentFolder(outputInternalPath_);

  auto bookSet = [&](const std::string& name, MassHistos& h, bool splitEta) {
    h.full = ibooker.book1D(
        name + "_mass", name + " opposite-charge invariant mass;M [GeV];Events", massBins_, massMin_, massMax_);
    h.zwin = ibooker.book1D(name + "_zMass", name + " Z window;M [GeV];Events", massBins_, zMin_, zMax_);

    if (splitEta) {
      h.barrel = ibooker.book1D(name + "_barrelMass", name + " barrel;M [GeV];Events", massBins_, massMin_, massMax_);
      h.endcap = ibooker.book1D(name + "_endcapMass", name + " endcap;M [GeV];Events", massBins_, massMin_, massMax_);
    }
  };

  if (doMuons_)
    bookSet("muons", muonHistos_, false);

  if (doElectrons_)
    bookSet("electrons", electronHistos_, true);
}

void ScoutingDileptonMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const&) {
  if (doMuons_) {
    edm::LogInfo("ScoutingDileptonMonitor") << "doing muons: ";
    edm::LogInfo("ScoutingDileptonMonitor") << "\n"
                                            << "MUON ID ::: " << muonID_;
    analyzeCollection(iEvent, muonToken_, muonCut_, muonHistos_, false, muonID_);
  }

  if (doElectrons_) {
    edm::LogInfo("ScoutingDileptonMonitor") << "doing electrons: ";
    analyzeCollection(iEvent, electronToken_, electronCut_, electronHistos_, true, electronID_);
  }
}

// ------------------------------------------------------------------------

template <typename T>
void ScoutingDileptonMonitor::analyzeCollection(const edm::Event& iEvent,
                                                const edm::EDGetTokenT<std::vector<T>>& token,
                                                const StringCutObjectSelector<T>& cut,
                                                MassHistos& histos,
                                                bool doEtaSplit,
                                                bool useID) {
  edm::LogInfo("ScoutingDileptonMonitor") << "IN ANALYZE COLLECTION";

  edm::Handle<std::vector<T>> handle;
  iEvent.getByToken(token, handle);

  if (!handle.isValid()) {
    edm::LogWarning("ScoutingDileptonMonitor") << "Invalid Handle!";
    return;
  } else {
    edm::LogInfo("ScoutingDileptonMonitor") << "Valid Handle!";
  }

  std::vector<const T*> selected;
  selected.reserve(handle->size());

  edm::LogInfo("ScoutingDileptonMonitor") << "collection size: " << handle->size();

  for (const auto& obj : *handle) {
    if (cut(obj)) {
      continue;
    }
    if (useID && !checkScoutingID(obj)) {
      continue;
    }
    selected.push_back(&obj);
  }

  fillPairs(selected, histos, doEtaSplit);
}

template <typename T>
void ScoutingDileptonMonitor::fillPairs(const std::vector<const T*>& leptons, MassHistos& histos, bool doEtaSplit) {
  const double massHypothesis =
      std::is_same_v<T, Run3ScoutingMuon> ? scoutingDQMUtils::MUON_MASS : scoutingDQMUtils::ELECTRON_MASS;

  const size_t n = leptons.size();
  edm::LogInfo("ScoutingDileptonMonitor") << "lepton size: " << n;

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      if (scouting::charge(*leptons[i]) * scouting::charge(*leptons[j]) >= 0)
        continue;

      const auto p4 = scouting::p4(*leptons[i], massHypothesis) + scouting::p4(*leptons[j], massHypothesis);

      const double mass = p4.mass();

      histos.full->Fill(mass);

      if (mass > zMin_ && mass < zMax_)
        histos.zwin->Fill(mass);

      if (doEtaSplit) {
        const bool barrel = std::abs(leptons[i]->eta()) < barrelEta_ && std::abs(leptons[j]->eta()) < barrelEta_;

        if (barrel)
          histos.barrel->Fill(mass);
        else
          histos.endcap->Fill(mass);
      }
    }
  }
}

void ScoutingDileptonMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("OutputInternalPath", "HLT/ScoutingOffline/DiLepton");
  desc.add<edm::InputTag>("muons", edm::InputTag("hltScoutingMuonPackerVtx"));
  desc.add<edm::InputTag>("electrons", edm::InputTag("hltScoutingEgammaPacker"));
  desc.add<bool>("doMuons", true);
  desc.add<bool>("doElectrons", true);
  desc.add<std::string>("muonCut", "");
  desc.add<std::string>("electronCut", "");
  desc.add<int>("massBins", 120);
  desc.add<double>("massMin", 0.0);
  desc.add<double>("massMax", 200.0);
  desc.add<double>("zMassMin", 70.0);
  desc.add<double>("zMassMax", 110.0);
  desc.add<double>("barrelEta", 1.479);
  desc.add<bool>("muonID", true);
  desc.add<bool>("electronID", true);
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(ScoutingDileptonMonitor);
