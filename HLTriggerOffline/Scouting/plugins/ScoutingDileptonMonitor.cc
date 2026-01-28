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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace scouting {

  inline int charge(const Run3ScoutingMuon& mu) {
    return mu.charge();
  }

  inline int charge(const Run3ScoutingElectron& el) {
    return el.trkcharge()[0];
  }

  template <typename T>
  math::PtEtaPhiMLorentzVector p4(const T& obj, double mass) {
    return math::PtEtaPhiMLorentzVector(obj.pt(), obj.eta(), obj.phi(), mass);
  }
}

class ScoutingDileptonMonitor : public DQMEDAnalyzer {
public:
  explicit ScoutingDileptonMonitor(const edm::ParameterSet&);
  ~ScoutingDileptonMonitor() override = default;

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
                         bool doEtaSplit);

  template <typename T>
  void fillPairs(const std::vector<const T*>&,
                 MassHistos&,
                 bool doEtaSplit);

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

  // ---- electrons --------------------------------------------------------
  const bool doElectrons_;
  const edm::EDGetTokenT<std::vector<Run3ScoutingElectron>> electronToken_;
  const StringCutObjectSelector<Run3ScoutingElectron> electronCut_;
  MassHistos electronHistos_;
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

      doElectrons_(iConfig.getParameter<bool>("doElectrons")),
      electronToken_(consumes<std::vector<Run3ScoutingElectron>>(iConfig.getParameter<edm::InputTag>("electrons"))),
      electronCut_(iConfig.getParameter<std::string>("electronCut")) {}

void ScoutingDileptonMonitor::bookHistograms(DQMStore::IBooker& ibooker,
                                             edm::Run const&,
                                             edm::EventSetup const&) {
  ibooker.setCurrentFolder(outputInternalPath_);

  auto bookSet = [&](const std::string& name, MassHistos& h, bool splitEta) {
    h.full = ibooker.book1D(
        name + "_mass",
        name + " opposite-charge invariant mass;M [GeV];Events",
        massBins_, massMin_, massMax_);

    h.zwin = ibooker.book1D(
        name + "_zMass",
        name + " Z window;M [GeV];Events",
        massBins_, zMin_, zMax_);

    if (splitEta) {
      h.barrel = ibooker.book1D(
          name + "_barrelMass",
          name + " barrel;M [GeV];Events",
          massBins_, massMin_, massMax_);

      h.endcap = ibooker.book1D(
          name + "_endcapMass",
          name + " endcap;M [GeV];Events",
          massBins_, massMin_, massMax_);
    }
  };

  if (doMuons_)
    bookSet("muons", muonHistos_, false);

  if (doElectrons_)
    bookSet("electrons", electronHistos_, true);
}

void ScoutingDileptonMonitor::analyze(edm::Event const& iEvent,
                                      edm::EventSetup const&) {
  if (doMuons_) {
    analyzeCollection(iEvent, muonToken_, muonCut_, muonHistos_, false);
  }

  if (doElectrons_) {
    analyzeCollection(iEvent, electronToken_, electronCut_, electronHistos_, true);
  }
}

// ------------------------------------------------------------------------

template <typename T>
void ScoutingDileptonMonitor::analyzeCollection(
    const edm::Event& iEvent,
    const edm::EDGetTokenT<std::vector<T>>& token,
    const StringCutObjectSelector<T>& cut,
    MassHistos& histos,
    bool doEtaSplit) {

  edm::Handle<std::vector<T>> handle;
  iEvent.getByToken(token, handle);
  if (!handle.isValid())
    return;

  std::vector<const T*> selected;
  selected.reserve(handle->size());

  for (const auto& obj : *handle) {
    if (cut(obj))
      selected.push_back(&obj);
  }

  fillPairs(selected, histos, doEtaSplit);
}

template <typename T>
void ScoutingDileptonMonitor::fillPairs(
    const std::vector<const T*>& leptons,
    MassHistos& histos,
    bool doEtaSplit) {

  constexpr double muMass = 0.105658;
  constexpr double elMass = 0.000511;

  const double massHypothesis =
      std::is_same_v<T, Run3ScoutingMuon> ? muMass : elMass;

  const size_t n = leptons.size();
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {

      if (scouting::charge(*leptons[i]) *
              scouting::charge(*leptons[j]) >= 0)
        continue;

      const auto p4 =
          scouting::p4(*leptons[i], massHypothesis) +
          scouting::p4(*leptons[j], massHypothesis);

      const double mass = p4.mass();

      histos.full->Fill(mass);

      if (mass > zMin_ && mass < zMax_)
        histos.zwin->Fill(mass);

      if (doEtaSplit) {
        const bool barrel =
            std::abs(leptons[i]->eta()) < barrelEta_ &&
            std::abs(leptons[j]->eta()) < barrelEta_;

        if (barrel)
          histos.barrel->Fill(mass);
        else
          histos.endcap->Fill(mass);
      }
    }
  }
}



DEFINE_FWK_MODULE(ScoutingDileptonMonitor);
