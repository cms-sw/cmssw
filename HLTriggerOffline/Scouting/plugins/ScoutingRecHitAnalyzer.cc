// system include files
#include <memory>

// FW include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// DQM include files
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

// Trigger bits
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"

// cut and function
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

// Dataformats
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/Scouting/interface/Run3ScoutingEBRecHit.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/Scouting/interface/Run3ScoutingHBHERecHit.h"

// C++ include files
#include <cmath>
#include <numeric>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>
#include <boost/algorithm/string/join.hpp>

// Trigger bits utlities
namespace util {
  bool isAnyTriggerAccept(const std::vector<std::string>& triggers,
                          const edm::TriggerResults& l1TriggerResults,
                          const edm::TriggerResultsByName& hltTriggerResultsByName) {
    for (auto const& trigger : triggers) {
      if (trigger.starts_with("L1_")) {
        const auto& trigger_names = l1TriggerResults.getTriggerNames();
        for (size_t itrigger = 0; itrigger < trigger_names.size(); itrigger++) {
          const std::string& trigger_name = trigger_names[itrigger];
          if (trigger_name.compare(0, trigger.length(), trigger) == 0) {
            if (l1TriggerResults.accept(itrigger))
              return true;
          }
        }
      } else {
        for (size_t itrigger = 0; itrigger < hltTriggerResultsByName.size(); itrigger++) {
          const std::string& trigger_name = hltTriggerResultsByName.triggerName(itrigger);
          if (trigger_name.compare(0, trigger.length() + 2, trigger + "_v") == 0) {  // ignore _v*
            if (hltTriggerResultsByName.accept(itrigger))
              return true;
          }
        }
      }
    }
    return false;
  }
}  // namespace util

template <typename RecHitType>
class ScoutingRecHitAnalyzer : public DQMEDAnalyzer {
public:
  using RecHitCollection = std::vector<RecHitType>;

  ScoutingRecHitAnalyzer(const edm::ParameterSet&);
  ~ScoutingRecHitAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void decodeDetId(const RecHitType& rechit, int& ieta, float& eta, int& iphi, float& phi);

  // input tokens
  const edm::EDGetTokenT<RecHitCollection> rechit_collection_token_;

  // trigger tokens
  const edm::EDGetTokenT<edm::TriggerResults> l1TriggerResults_token_;
  const edm::EDGetTokenT<edm::TriggerResults> hltTriggerResults_token_;

  // functor
  const StringCutObjectSelector<RecHitType> cut_;  // general cut applied to all object

  // other parameters
  const std::string topFolderName_;  // top folder name where to book histograms

  // triggers
  std::vector<std::vector<std::string>> triggers_;  // trigger expressions

  // histograms
  std::vector<MonitorElement*> number_histograms_;
  std::vector<MonitorElement*> energy_histograms_;
  std::vector<MonitorElement*> time_histograms_;
  std::vector<MonitorElement*> ieta_iphi_histograms_;
  std::vector<MonitorElement*> eta_phi_histograms_;
  std::vector<MonitorElement*> energy_time_histograms_;
  std::vector<MonitorElement*> ieta_iphi_energy_profiles_;
  std::vector<MonitorElement*> eta_phi_energy_profiles_;
  std::vector<MonitorElement*> ieta_iphi_time_profiles_;
  std::vector<MonitorElement*> eta_phi_time_profiles_;

  // trigger names
  std::vector<std::string> trigger_names_;

  // named constants
  static constexpr double kDegToRad = M_PI / 180.0;
  static constexpr double kFiveDegToRad = 5.0 * M_PI / 180.0;
};

template <typename RecHitType>
ScoutingRecHitAnalyzer<RecHitType>::ScoutingRecHitAnalyzer(const edm::ParameterSet& iConfig)
    : rechit_collection_token_(consumes(iConfig.getParameter<edm::InputTag>("src"))),
      l1TriggerResults_token_(consumes(iConfig.getParameter<edm::InputTag>("L1TriggerResults"))),
      hltTriggerResults_token_(consumes(iConfig.getParameter<edm::InputTag>("HLTTriggerResults"))),
      cut_(iConfig.getParameter<std::string>("cut"), iConfig.getUntrackedParameter<bool>("lazy_eval")),
      topFolderName_(iConfig.getParameter<std::string>("topFolderName")) {
  static_assert(
      std::is_same<RecHitType, Run3ScoutingEBRecHit>::value || std::is_same<RecHitType, Run3ScoutingHBHERecHit>::value,
      "Unsupported Type of RecHit");

  std::string no_trigger_name = "No_Trigger";
  trigger_names_.push_back(no_trigger_name);

  const auto& trigger_vpset = iConfig.getParameter<std::vector<edm::ParameterSet>>("triggers");
  for (auto const& trigger_pset : trigger_vpset) {
    std::vector<std::string> trigger = trigger_pset.getParameter<std::vector<std::string>>("expr");
    std::string trigger_name = trigger_pset.getParameter<std::string>("name");
    if (trigger_name.empty())
      trigger_name = boost::algorithm::join(trigger, "-or-");
    triggers_.push_back(trigger);
    trigger_names_.push_back(trigger_name);
  }
}

template <typename RecHitType>
void ScoutingRecHitAnalyzer<RecHitType>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src");
  desc.add<std::string>("topFolderName", "HLT/ScoutingOffline/CaloRecHits");
  desc.add<edm::InputTag>("L1TriggerResults", edm::InputTag("l1bits"));
  desc.add<edm::InputTag>("HLTTriggerResults", edm::InputTag("TriggerResults", "", "HLT"));
  desc.addUntracked<bool>("lazy_eval", false);
  desc.add<std::string>("cut", "");

  edm::ParameterSetDescription trigger_desc;
  trigger_desc.add<std::string>("name", "");
  trigger_desc.add<std::vector<std::string>>("expr");
  edm::ParameterSet trigger_pset;
  trigger_pset.addParameter<std::string>("name", "");
  trigger_pset.addParameter<std::vector<std::string>>("expr", {"DST_PFScouting_JetHT"});
  std::vector<edm::ParameterSet> trigger_vpset;
  trigger_vpset.push_back(trigger_pset);
  desc.addVPSet("triggers", trigger_desc, trigger_vpset);

  descriptions.addWithDefaultLabel(desc);
}

template <typename RecHitType>
void ScoutingRecHitAnalyzer<RecHitType>::bookHistograms(DQMStore::IBooker& ibooker,
                                                        edm::Run const&,
                                                        edm::EventSetup const&) {
  ibooker.setCurrentFolder(topFolderName_);
  for (auto const& trigger_name : trigger_names_) {
    ibooker.setCurrentFolder(topFolderName_ + "/" + trigger_name);
    if constexpr (std::is_same<RecHitType, Run3ScoutingEBRecHit>()) {
      number_histograms_.push_back(ibooker.book1D("number", "Number;Events", 100, 0., 1000.));
      energy_histograms_.push_back(ibooker.book1D("energy", "Energy (GeV);Events", 100, 0., 20.));
      time_histograms_.push_back(ibooker.book1D("time", "Time (ps);Events", 100, 0., 1000.));
      energy_time_histograms_.push_back(
          ibooker.book2D("energy_time", "Energy (GeV);Time (ps);Entries", 100, 0., 20., 100, 0., 1000.));
      ieta_iphi_histograms_.push_back(
          ibooker.book2D("ieta_iphi", "i#eta;i#phi;Entries", 171, -85.5, 85.5, 360, 0.5, 360.5));
      eta_phi_histograms_.push_back(
          ibooker.book2D("eta_phi", "#eta;#phi;Entries", 170, -85 * kDegToRad, 85 * kDegToRad, 360, -M_PI, M_PI));
      ieta_iphi_energy_profiles_.push_back(ibooker.bookProfile2D(
          "ieta_iphi_energy", "i#eta;i#phi;mean Energy", 171, -85.5, 85.5, 360, 0.5, 360.5, 0, 20));
      eta_phi_energy_profiles_.push_back(ibooker.bookProfile2D(
          "eta_phi_energy", "#eta;#phi;mean Energy", 170, -85 * kDegToRad, 85 * kDegToRad, 360, -M_PI, M_PI, 0, 20));

      ieta_iphi_time_profiles_.push_back(
          ibooker.bookProfile2D("ieta_iphi_time", "i#eta;i#phi;mean Time", 171, -85.5, 85.5, 360, 0.5, 360.5, 0, 20));
      eta_phi_time_profiles_.push_back(ibooker.bookProfile2D(
          "eta_phi_time", "#eta;#phi;mean Time", 170, -85 * kDegToRad, 85 * kDegToRad, 360, -M_PI, M_PI, 0, 20));

    } else if constexpr (std::is_same<RecHitType, Run3ScoutingHBHERecHit>()) {
      std::vector<double> eta_edges(59);
      eta_edges[29] = 0.0;
      double deta = 0.087266462599716475;
      double eta_edge = 0;
      for (unsigned int ieta = 1; ieta <= 29; ieta++) {
        if (ieta == 21)
          deta *= 2;
        eta_edge += deta;
        eta_edges[29 + ieta] = eta_edge;
        eta_edges[29 - ieta] = -1 * eta_edge;
      }

      // Build uniform phi bin edges
      int nPhiBins = 72;
      std::vector<double> phi_edges(nPhiBins + 1);
      double phiMin = -M_PI;
      double phiMax = M_PI;
      double dphi = (phiMax - phiMin) / nPhiBins;
      std::iota(phi_edges.begin(), phi_edges.end(), 0);  // fill with 0,1,2,...
      std::transform(phi_edges.begin(), phi_edges.end(), phi_edges.begin(), [phiMin, dphi](double i) {
        return phiMin + i * dphi;
      });

      number_histograms_.push_back(ibooker.book1D("number", "Number;Events", 100, 0., 2000.));
      energy_histograms_.push_back(ibooker.book1D("energy", "Energy (GeV);Events", 100, 0., 20.));
      time_histograms_.push_back(ibooker.book1D("time", "Time (ns);Events", 100, 0., 30.));
      energy_time_histograms_.push_back(
          ibooker.book2D("energy_time", "Energy (GeV);Time (ns);Entries", 100., 0., 20., 100., 0., 30.));

      ieta_iphi_histograms_.push_back(
          ibooker.book2D("ieta_iphi", "i#eta;i#phi;Entries", 59, -29.5, 29.5, 72, 0.5, 72.5));
      ieta_iphi_energy_profiles_.push_back(
          ibooker.bookProfile2D("ieta_iphi_energy", "i#eta;i#phi;mean Energy", 59, -29.5, 29.5, 72, 0.5, 72.5, 0, 20));

      ieta_iphi_time_profiles_.push_back(
          ibooker.bookProfile2D("ieta_iphi_time", "i#eta;i#phi;mean Time", 59, -29.5, 29.5, 72, 0.5, 72.5, 0, 20));

      // demote edges from double to float
      std::vector<float> eta_edges_f(eta_edges.begin(), eta_edges.end());
      std::vector<float> phi_edges_f(phi_edges.begin(), phi_edges.end());
      eta_phi_histograms_.push_back(ibooker.book2D("eta_phi",
                                                   "#eta;#phi;Entries",
                                                   eta_edges_f.size() - 1,
                                                   eta_edges_f.data(),
                                                   phi_edges_f.size() - 1,
                                                   phi_edges_f.data()));

      eta_phi_energy_profiles_.push_back(ibooker.bookProfile2D("eta_phi_energy",
                                                               "#eta;#phi;mean Energy",
                                                               eta_edges.size() - 1,
                                                               eta_edges.data(),
                                                               phi_edges.size() - 1,
                                                               phi_edges.data()));

      eta_phi_time_profiles_.push_back(ibooker.bookProfile2D("eta_phi_time",
                                                             "#eta;#phi;mean Time",
                                                             eta_edges.size() - 1,
                                                             eta_edges.data(),
                                                             phi_edges.size() - 1,
                                                             phi_edges.data()));
    }
  }
}

template <typename RecHitType>
void ScoutingRecHitAnalyzer<RecHitType>::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& l1TriggerResults_handle = iEvent.getHandle(l1TriggerResults_token_);
  if (!l1TriggerResults_handle.isValid()) {
    edm::LogWarning("Handle") << "L1 TriggerResults is invalid";
    return;
  }

  const auto& hltTriggerResults_handle = iEvent.getHandle(hltTriggerResults_token_);
  if (!hltTriggerResults_handle.isValid()) {
    edm::LogWarning("Handle") << "HLT TriggerResults is invalid";
    return;
  }

  const auto& hltTriggerResultsByName = iEvent.triggerResultsByName(*hltTriggerResults_handle);

  const auto& rechit_collection_handle = iEvent.getHandle(rechit_collection_token_);
  if (!rechit_collection_handle.isValid()) {
    edm::LogWarning("Handle") << "rechit is invalid";
    return;
  }

  std::vector<bool> trigger_decisions(triggers_.size(), false);
  for (unsigned int itrigger = 0; itrigger < triggers_.size(); itrigger++) {
    const auto& trigger = triggers_[itrigger];
    trigger_decisions[itrigger] = util::isAnyTriggerAccept(trigger, *l1TriggerResults_handle, hltTriggerResultsByName);
  }

  unsigned int number_passed = 0;

  for (unsigned int irechit = 0; irechit < rechit_collection_handle->size(); irechit++) {
    const auto& rechit = rechit_collection_handle->at(irechit);
    int ieta, iphi;
    float eta, phi;
    decodeDetId(rechit, ieta, eta, iphi, phi);
    const auto& energy = rechit.energy();
    const auto& time = rechit.time();

    if (cut_(rechit)) {
      energy_histograms_[0]->Fill(energy);
      time_histograms_[0]->Fill(time);
      ieta_iphi_histograms_[0]->Fill(ieta, iphi);
      eta_phi_histograms_[0]->Fill(eta, phi);
      energy_time_histograms_[0]->Fill(energy, time);
      ieta_iphi_energy_profiles_[0]->Fill(ieta, iphi, energy);
      eta_phi_energy_profiles_[0]->Fill(eta, phi, energy);
      ieta_iphi_time_profiles_[0]->Fill(ieta, iphi, time);
      eta_phi_time_profiles_[0]->Fill(eta, phi, time);
      if (std::is_same<RecHitType, Run3ScoutingHBHERecHit>()) {
        if (std::abs(ieta) >= 21) {
          float phi2 = (iphi + 0.5) * 5 * M_PI / 180.;
          if (phi2 > M_PI)
            phi2 = phi2 - 2 * M_PI;
          eta_phi_histograms_[0]->Fill(eta, phi2);
          eta_phi_energy_profiles_[0]->Fill(eta, phi2, energy);
          eta_phi_time_profiles_[0]->Fill(eta, phi2, time);
        }
      }
      unsigned int itrigger = 0;
      for (const bool& trigger_accept : trigger_decisions) {
        itrigger++;
        if (trigger_accept) {
          energy_histograms_[itrigger]->Fill(energy);
          time_histograms_[itrigger]->Fill(time);
          ieta_iphi_histograms_[itrigger]->Fill(ieta, iphi);
          eta_phi_histograms_[itrigger]->Fill(eta, phi);
          energy_time_histograms_[itrigger]->Fill(energy, time);
          ieta_iphi_energy_profiles_[itrigger]->Fill(ieta, iphi, energy);
          eta_phi_energy_profiles_[itrigger]->Fill(eta, phi, energy);
          ieta_iphi_time_profiles_[itrigger]->Fill(ieta, iphi, time);
          eta_phi_time_profiles_[itrigger]->Fill(eta, phi, time);
          if (std::is_same<RecHitType, Run3ScoutingHBHERecHit>()) {
            if (std::abs(ieta) >= 21) {
              float phi2 = (iphi + 0.5) * 5 * M_PI / 180.;
              if (phi2 > M_PI)
                phi2 = phi2 - 2 * M_PI;
              eta_phi_histograms_[itrigger]->Fill(eta, phi2);
              eta_phi_energy_profiles_[itrigger]->Fill(eta, phi2, energy);
              eta_phi_time_profiles_[itrigger]->Fill(eta, phi2, time);
            }
          }
        }
      }
      number_passed++;
    }
  }

  number_histograms_[0]->Fill(number_passed);
  unsigned int itrigger = 0;
  for (const bool& trigger_accept : trigger_decisions) {
    itrigger++;
    if (trigger_accept) {
      number_histograms_[itrigger]->Fill(number_passed);
    }
  }
}

template <typename RecHitType>
void ScoutingRecHitAnalyzer<RecHitType>::decodeDetId(
    const RecHitType& rechit, int& ieta, float& eta, int& iphi, float& phi) {
  if constexpr (std::is_same<RecHitType, Run3ScoutingEBRecHit>()) {
    const auto& detId = EBDetId(rechit.detId());
    ieta = detId.ieta();
    const auto& ietaAbs = detId.ietaAbs();
    const auto& zside = detId.zside();
    eta = zside * (ietaAbs - 0.5) * kDegToRad;
    iphi = detId.iphi();
    phi = (iphi - 0.5) * M_PI / 180.;
    if (phi > M_PI)
      phi = phi - 2 * M_PI;
  } else if constexpr (std::is_same<RecHitType, Run3ScoutingHBHERecHit>()) {
    const auto& detId = HcalDetId(rechit.detId());
    ieta = detId.ieta();
    const auto& ietaAbs = detId.ietaAbs();
    const auto& zside = detId.zside();
    if (ietaAbs <= 20) {
      eta = zside * (ietaAbs - 0.5) * kFiveDegToRad;
    } else {
      eta = zside * ((20 * kFiveDegToRad) + (ietaAbs - 20 - 0.5) * kFiveDegToRad * 2);
    }
    iphi = detId.iphi();
    phi = (iphi - 0.5) * 5 * M_PI / 180.;
    if (phi > M_PI)
      phi = phi - 2 * M_PI;
  }
}

using ScoutingEBRecHitAnalyzer = ScoutingRecHitAnalyzer<Run3ScoutingEBRecHit>;
DEFINE_FWK_MODULE(ScoutingEBRecHitAnalyzer);

using ScoutingHBHERecHitAnalyzer = ScoutingRecHitAnalyzer<Run3ScoutingHBHERecHit>;
DEFINE_FWK_MODULE(ScoutingHBHERecHitAnalyzer);
