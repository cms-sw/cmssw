// system includes
#include <vector>
#include <cmath>
#include <algorithm>

// user includes
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Scouting/interface/Run3ScoutingParticle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ScoutingPi0Analyzer : public DQMEDAnalyzer {
  // Helper struct to hold all histograms for a single collection (Scouting or Packed)
  struct Pi0Histograms {
    // Kinematics: Input Photons (before selection)
    MonitorElement* h_input_pt;
    MonitorElement* h_input_eta;
    MonitorElement* h_input_phi;

    // Kinematics: Selected Photons (after isolation)
    MonitorElement* h_sel_pt;
    MonitorElement* h_sel_eta;
    MonitorElement* h_sel_phi;

    // Cut Variables: Isolation
    MonitorElement* h_iso_maxPtRatio;  // Max (Pt_hadron / Pt_photon) in cone
    MonitorElement* h_iso_minDr;       // Min dR to hadron

    // Cut Variables: Pairs
    MonitorElement* h_pair_pt;    // Pair Pt
    MonitorElement* h_pair_asym;  // Energy Asymmetry
    MonitorElement* h_pair_dr;    // Delta R between photons

    // Selected photons plots
    MonitorElement* h_selpair_mass;        // Final Invariant Mass
    MonitorElement* h_selpair_pt;          // Final candidate pT
    MonitorElement* h_selpair_leadPt;      // Final candidate leading pT
    MonitorElement* h_selpair_subleadPt;   // Final candidate sub-lead pT
    MonitorElement* h_selpair_leadEta;     // Final candidate leading eta
    MonitorElement* h_selpair_subleadEta;  // Final candidate sub-lead eta
  };

public:
  explicit ScoutingPi0Analyzer(const edm::ParameterSet&);
  ~ScoutingPi0Analyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // Helper to book a set of histograms
  void bookHistSet(DQMStore::IBooker& ibook, Pi0Histograms& hists, const std::string& suffix);
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // Templated analysis function to handle both Scouting and PAT types
  template <typename T>
  void analyzeCollection(const edm::Handle<std::vector<T>>& handle, MonitorElement* hist);

  // Tokens
  const edm::EDGetTokenT<std::vector<Run3ScoutingParticle>> scoutingToken_;

  // Configuration parameters
  const std::string outputInternalPath_;
  const double minPt_;
  const double maxEta_;
  const double isolationConeSq_;
  const double isolationPtRatio_;
  const double pairMaxDrSq_;
  const double asymmetryCut_;
  const double pairMinPt_;
  const double maxMass_;

  // Histograms
  Pi0Histograms h_scouting;
  MonitorElement* h_mass_scouting;
};

ScoutingPi0Analyzer::ScoutingPi0Analyzer(const edm::ParameterSet& iConfig)
    : scoutingToken_(
          consumes<std::vector<Run3ScoutingParticle>>(iConfig.getParameter<edm::InputTag>("scoutingCollection"))),
      outputInternalPath_{iConfig.getParameter<std::string>("OutputInternalPath")},
      minPt_(iConfig.getParameter<double>("minPt")),
      maxEta_(iConfig.getParameter<double>("maxEta")),
      isolationConeSq_(std::pow(iConfig.getParameter<double>("isolationCone"), 2)),
      isolationPtRatio_(iConfig.getParameter<double>("isolationPtRatio")),
      pairMaxDrSq_(std::pow(iConfig.getParameter<double>("pairMaxDr"), 2)),
      asymmetryCut_(iConfig.getParameter<double>("asymmetryCut")),
      pairMinPt_(iConfig.getParameter<double>("pairMinPt")),
      maxMass_(iConfig.getParameter<double>("maxMass")) {}

void ScoutingPi0Analyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("scoutingCollection", edm::InputTag("hltScoutingPFPacker"));
  desc.add<std::string>("OutputInternalPath", "HLT/ScoutingOffline/PiZero");
  desc.add<double>("minPt", 1.5)->setComment("minimum pT of input photons");
  desc.add<double>("maxEta", 2.5)->setComment("maximum eta of input photons");
  desc.add<double>("isolationCone", 0.2)->setComment("isolation cone radius");  // will be squared internally
  desc.add<double>("isolationPtRatio", 0.8)->setComment("charged hadron isolation ratio");
  desc.add<double>("pairMaxDr", 0.1)->setComment("maximum DeltaR between photons");  // will be squared internally
  desc.add<double>("asymmetryCut", 0.85)->setComment("Energy asymmetry cut value");
  desc.add<double>("pairMinPt", 2.0)->setComment("mimimum pT of the di-photon");
  desc.add<double>("maxMass", 1.)->setComment("maximum invariant mass accepted");
  descriptions.addWithDefaultLabel(desc);
}

void ScoutingPi0Analyzer::bookHistSet(DQMStore::IBooker& ibook, Pi0Histograms& h, const std::string& suffix) {
  // Input Kinematics
  h.h_input_pt = ibook.book1D(("h_input_pt_" + suffix).c_str(),
                              ("Input Photon p_{T} (" + suffix + ")#gamma ;p_{T} [GeV];Entries").c_str(),
                              50,
                              0,
                              20);
  h.h_input_eta = ibook.book1D(("h_input_eta_" + suffix).c_str(),
                               ("Input Photon #eta (" + suffix + ")#gamma ;#eta;Entries").c_str(),
                               60,
                               -3.0,
                               3.0);
  h.h_input_phi = ibook.book1D(("h_input_phi_" + suffix).c_str(),
                               ("Input Photon #phi (" + suffix + ");#gamma #phi;Entries").c_str(),
                               64,
                               -3.2,
                               3.2);

  // Selected Kinematics
  h.h_sel_pt = ibook.book1D(("h_sel_pt_" + suffix).c_str(),
                            ("Selected Photon p_{T} (" + suffix + ");#gamma p_{T} [GeV];Entries").c_str(),
                            50,
                            0,
                            20);
  h.h_sel_eta = ibook.book1D(("h_sel_eta_" + suffix).c_str(),
                             ("Selected Photon #eta (" + suffix + ");#gamma #eta;Entries").c_str(),
                             60,
                             -3.0,
                             3.0);
  h.h_sel_phi = ibook.book1D(("h_sel_phi_" + suffix).c_str(),
                             ("Selected Photon #phi (" + suffix + ");#gamma #phi;Entries").c_str(),
                             64,
                             -3.2,
                             3.2);

  // Cut Variables: Isolation
  h.h_iso_maxPtRatio = ibook.book1D(("h_iso_maxPtRatio_" + suffix).c_str(),
                                    ("Max Hadron p_{T} / Photon p_{T} in Cone (" + suffix + ");Ratio;Entries").c_str(),
                                    50,
                                    0,
                                    2.0);
  h.h_iso_minDr = ibook.book1D(("h_iso_minDr_" + suffix).c_str(),
                               ("Min #Delta R #gamma to Hadron (" + suffix + ");#Delta R(#gamma,had);Entries").c_str(),
                               50,
                               0,
                               0.5);

  // Cut Variables: Pairs
  h.h_pair_pt = ibook.book1D(("h_pair_pt_" + suffix).c_str(),
                             ("Diphoton p_{T} (" + suffix + ");#gamma#gamma p_{T} [GeV];Entries").c_str(),
                             50,
                             0,
                             50);
  h.h_pair_asym = ibook.book1D(
      ("h_pair_asym_" + suffix).c_str(),
      ("Diphoton Energy Asymmetry (" + suffix + ");|E_{#gamma 1}-E_{#gamma 2}|/(E_{#gamma 1}+E_{#gamma 2});Entries")
          .c_str(),
      50,
      0,
      1.0);
  h.h_pair_dr = ibook.book1D(("h_pair_dr_" + suffix).c_str(),
                             ("Diphoton #Delta R (" + suffix + ");#Delta R(#gamma,#gamma);Entries").c_str(),
                             50,
                             0,
                             1.0);

  // Selected final di-photons plots
  h.h_selpair_mass = ibook.book1D(("h_mass_" + suffix).c_str(),
                                  ("Diphoton Invariant Mass (" + suffix + ");M_{#gamma#gamma} [GeV];Entries").c_str(),
                                  100,
                                  0,
                                  1.0);

  h.h_selpair_pt = ibook.book1D(("h_selpair_pt_" + suffix).c_str(),
                                ("Selected Diphoton p_{T} (" + suffix + ");#gamma#gamma p_{T} [GeV];Entries").c_str(),
                                50,
                                0,
                                50);

  // Selected final photons kinematics
  h.h_selpair_leadPt =
      ibook.book1D(("h_selpair_leadPt_" + suffix).c_str(),
                   ("Leading Photon p_{T} (" + suffix + ");leading #gamma p_{T} [GeV];Entries").c_str(),
                   50,
                   0,
                   50);
  h.h_selpair_leadEta = ibook.book1D(("h_selpair_leadEta_" + suffix).c_str(),
                                     ("Leading Photon #eta (" + suffix + ");leading #gamma #eta;Entries").c_str(),
                                     60,
                                     -3.0,
                                     3.0);

  h.h_selpair_subleadPt =
      ibook.book1D(("h_selpair_subleadPt_" + suffix).c_str(),
                   ("Subleading Photon p_{T} (" + suffix + ");subleading #gamma p_{T} [GeV];Entries").c_str(),
                   50,
                   0,
                   50);
  h.h_selpair_subleadEta =
      ibook.book1D(("h_selpair_subleadEta_" + suffix).c_str(),
                   ("Subleading Photon #eta (" + suffix + ");subleading #gamma #eta;Entries").c_str(),
                   60,
                   -3.0,
                   3.0);
}

void ScoutingPi0Analyzer::bookHistograms(DQMStore::IBooker& ibook, edm::Run const&, edm::EventSetup const&) {
  ibook.setCurrentFolder(outputInternalPath_);

  // Scouting Histogram
  h_mass_scouting =
      ibook.book1D("h_mass_scouting", "Diphoton Mass (Scouting);Invariant Mass [GeV];Entries", 100, 0.0, maxMass_);

  bookHistSet(ibook, h_scouting, "Scouting");
}

void ScoutingPi0Analyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // 1. Analyze Scouting Particles
  edm::Handle<std::vector<Run3ScoutingParticle>> scoutingHandle;
  iEvent.getByToken(scoutingToken_, scoutingHandle);
  if (scoutingHandle.isValid()) {
    analyzeCollection(scoutingHandle, h_mass_scouting);
  }
}

template <typename T>
void ScoutingPi0Analyzer::analyzeCollection(const edm::Handle<std::vector<T>>& handle, MonitorElement* hist) {
  std::vector<const T*> photons;
  std::vector<const T*> hadrons;
  photons.reserve(handle->size());
  hadrons.reserve(handle->size());

  // 1. Selection and Categorization
  for (const auto& cand : *handle) {
    if (cand.pt() < minPt_)
      continue;
    if (std::abs(cand.eta()) > maxEta_)
      continue;

    int pdgId = std::abs(cand.pdgId());

    if (pdgId == 22) {  // Photon
      photons.push_back(&cand);

      // Diagnostic: Input Kinematics
      h_scouting.h_input_pt->Fill(cand.pt());
      h_scouting.h_input_eta->Fill(cand.eta());
      h_scouting.h_input_phi->Fill(cand.phi());

    } else if (pdgId == 211 || pdgId == 130 || pdgId == 321 || pdgId == 2212) {  // Hadrons
      hadrons.push_back(&cand);
    }
  }

  // 2. Isolation Logic
  std::vector<const T*> good_photons;
  good_photons.reserve(photons.size());

  for (const auto* g : photons) {
    bool is_isolated = true;
    double max_pt_ratio = -1.0;
    double min_dr = 999.0;

    for (const auto* h : hadrons) {
      double dr2 = reco::deltaR2(*g, *h);
      double dr = std::sqrt(dr2);

      // Track closest hadron for diagnostics
      if (dr < min_dr)
        min_dr = dr;

      // Isolation Check
      if (dr2 < isolationConeSq_) {
        double ratio = h->pt() / g->pt();
        if (ratio > max_pt_ratio)
          max_pt_ratio = ratio;  // Track worst ratio

        if (ratio > isolationPtRatio_) {
          is_isolated = false;
          // Note: We don't break here so we can find the true max_pt_ratio for plotting
        }
      }
    }

    // Diagnostic: Isolation variables (N-1 style not fully possible without loop overhead,
    // but we plot the "worst offender" variables)
    if (max_pt_ratio >= 0)
      h_scouting.h_iso_maxPtRatio->Fill(max_pt_ratio);
    if (min_dr < 999)
      h_scouting.h_iso_minDr->Fill(min_dr);

    if (is_isolated) {
      good_photons.push_back(g);
      // Diagnostic: Selected Kinematics
      h_scouting.h_sel_pt->Fill(g->pt());
      h_scouting.h_sel_eta->Fill(g->eta());
      h_scouting.h_sel_phi->Fill(g->phi());
    }
  }

  // 3. Combinatorics
  // Logic from script: limit to 30 good photons to prevent combinatorial explosion
  size_t n_good = std::min(good_photons.size(), size_t(100));

  for (size_t i = 0; i < n_good; ++i) {
    const auto* g1 = good_photons[i];

    for (size_t j = i + 1; j < n_good; ++j) {
      const auto* g2 = good_photons[j];

      // Manual p4 reconstruction to match script's "mass=0" assumption for photons
      // (Run3ScoutingParticle has mass, but we treat it as photon here)
      math::PtEtaPhiMLorentzVector p4_1(g1->pt(), g1->eta(), g1->phi(), 0.0);
      math::PtEtaPhiMLorentzVector p4_2(g2->pt(), g2->eta(), g2->phi(), 0.0);

      auto p4_gg = p4_1 + p4_2;

      double E1 = p4_1.energy();
      double E2 = p4_2.energy();
      double asym = std::abs(E1 - E2) / (E1 + E2);
      double dr_gg = reco::deltaR(*g1, *g2);

      // Diagnostic: Fill variables before pair cuts
      h_scouting.h_pair_pt->Fill(p4_gg.pt());
      h_scouting.h_pair_asym->Fill(asym);
      h_scouting.h_pair_dr->Fill(dr_gg);

      // Cuts
      if (p4_gg.pt() < pairMinPt_)
        continue;
      if (asym > asymmetryCut_)
        continue;
      if (dr_gg * dr_gg > pairMaxDrSq_)
        continue;  // Note: script used 0.5^2
      if (p4_gg.mass() > maxMass_)
        continue;

      h_scouting.h_selpair_mass->Fill(p4_gg.mass());
      h_scouting.h_selpair_pt->Fill(p4_gg.pt());

      const auto& lead_p4 = (p4_1.pt() > p4_2.pt()) ? p4_1 : p4_2;
      const auto& sub_p4 = (p4_1.pt() > p4_2.pt()) ? p4_2 : p4_1;

      h_scouting.h_selpair_leadPt->Fill(lead_p4.pt());
      h_scouting.h_selpair_subleadPt->Fill(sub_p4.pt());

      h_scouting.h_selpair_leadEta->Fill(lead_p4.eta());
      h_scouting.h_selpair_subleadEta->Fill(sub_p4.eta());

      // Fill Histogram
      hist->Fill(p4_gg.mass());
    }
  }
}

DEFINE_FWK_MODULE(ScoutingPi0Analyzer);
