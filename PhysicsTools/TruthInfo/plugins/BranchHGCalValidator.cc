// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

// DQM performance plots for the truth::Branch graph as a replacement for the
// legacy HGCAL truth objects (CaloParticle, SimCluster / SimTracksters). For each
// legacy object it finds the logical Branch that should reproduce it (via the
// SimTrack trackId), compares the Branch's subgraph calo hits to the object's
// hits_and_fractions, and asks the generic BranchHitAssociator whether that same
// Branch is the best hit-based match. The booked numerator/denominator histograms
// are turned into a "reproduction efficiency vs eta/pt/energy" by the harvester
// (DQMGenericClient); purity, completeness and energy-response are booked
// directly. These sit alongside the standard SimTrackster/CaloParticle plots so
// the two truth descriptions can be compared in the same DQM output.

#include <algorithm>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"

#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/TruthGraph.h"

class BranchHGCalValidator : public DQMEDAnalyzer {
public:
  explicit BranchHGCalValidator(edm::ParameterSet const&);
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  // One set of monitor elements per legacy collection (CaloParticle, SimCluster).
  struct Plots {
    // Numerator/denominator for the harvester-computed reproduction efficiency.
    MonitorElement* denomEta = nullptr;
    MonitorElement* denomPt = nullptr;
    MonitorElement* denomEnergy = nullptr;
    MonitorElement* effNumEta = nullptr;
    MonitorElement* effNumPt = nullptr;
    MonitorElement* effNumEnergy = nullptr;
    // Quality distributions.
    MonitorElement* purity = nullptr;
    MonitorElement* completenessHits = nullptr;
    MonitorElement* completenessEnergy = nullptr;
    MonitorElement* energyResponse = nullptr;
    // Raw energy response: Branch hit energy over the object's *hit* energy (rather
    // than its generator energy), on the deposited (sim) and reconstructed (rec)
    // scales. ~1 when the Branch reproduces the object's calorimeter energy.
    MonitorElement* rawEnergyResponseSim = nullptr;
    MonitorElement* rawEnergyResponseReco = nullptr;
    // Profiles vs kinematics.
    MonitorElement* purityVsEta = nullptr;
    MonitorElement* completenessVsEta = nullptr;
    MonitorElement* responseVsEta = nullptr;
    MonitorElement* responseVsEnergy = nullptr;
    MonitorElement* rawResponseSimVsEnergy = nullptr;
    MonitorElement* rawResponseRecoVsEnergy = nullptr;

    // "Other way around": for each truth object, its best hit-matched Branch (which
    // need not be the natural, trackId-seeded one) and that Branch's performance.
    MonitorElement* bestPurity = nullptr;
    MonitorElement* bestCompletenessHits = nullptr;
    MonitorElement* bestCompletenessEnergy = nullptr;
    MonitorElement* bestResponse = nullptr;
    // Self-match numerator: best hit-matched Branch == natural Branch (denom reused).
    MonitorElement* selfMatchEta = nullptr;
    MonitorElement* selfMatchPt = nullptr;
    // Merge/split: distinct Branches sharing >=10% of the object's hits.
    MonitorElement* nSharingBranches = nullptr;
  };

  void book(DQMStore::IBooker&, Plots&, std::string const& sub);

  template <class Collection>
  void validate(Collection const& objects,
                truth::Graph const& graph,
                TruthGraph const& raw,
                truth::LogicalGraphHitIndex const& hitIndex,
                truth::BranchHitAssociator const& assoc,
                std::unordered_map<uint32_t, uint32_t> const& tidToParticle,
                std::unordered_map<uint32_t, float> const& cellSimEnergy,
                std::unordered_map<uint32_t, float> const& recHitEnergyByDetId,
                Plots& plots);

  // Whole-cell RecHit energy keyed by DetId, rebuilt from the same RecHit
  // collections (HGCal then PF, the DetIdToRecHitMapProducer order) the hit
  // index was mapped against; first index kept for any duplicate DetId.
  std::unordered_map<uint32_t, float> collectRecHitEnergyByDetId(edm::Event const&) const;

  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<TruthGraph> rawToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<CaloParticle>> caloParticleToken_;
  const edm::EDGetTokenT<std::vector<SimCluster>> simClusterToken_;
  std::vector<edm::EDGetTokenT<HGCRecHitCollection>> hgcalRecHitTokens_;
  std::vector<edm::EDGetTokenT<reco::PFRecHitCollection>> pfRecHitTokens_;
  std::vector<edm::InputTag> hgcalRecHitTags_;
  std::vector<edm::InputTag> pfRecHitTags_;

  const std::string folder_;
  const double minPt_;
  const double maxEta_;

  Plots caloParticlePlots_;
  Plots simClusterPlots_;
};

BranchHGCalValidator::BranchHGCalValidator(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      rawToken_(consumes<TruthGraph>(cfg.getParameter<edm::InputTag>("rawSrc"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      caloParticleToken_(consumes<std::vector<CaloParticle>>(cfg.getParameter<edm::InputTag>("caloParticles"))),
      simClusterToken_(consumes<std::vector<SimCluster>>(cfg.getParameter<edm::InputTag>("simClusters"))),
      folder_(cfg.getParameter<std::string>("folder")),
      minPt_(cfg.getParameter<double>("minPt")),
      maxEta_(cfg.getParameter<double>("maxEta")) {
  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("hgcalRecHits")) {
    hgcalRecHitTags_.push_back(tag);
    hgcalRecHitTokens_.push_back(consumes<HGCRecHitCollection>(tag));
  }
  for (auto const& tag : cfg.getParameter<std::vector<edm::InputTag>>("pfRecHits")) {
    pfRecHitTags_.push_back(tag);
    pfRecHitTokens_.push_back(consumes<reco::PFRecHitCollection>(tag));
  }
}

void BranchHGCalValidator::book(DQMStore::IBooker& ib, Plots& p, std::string const& sub) {
  ib.setCurrentFolder(folder_ + "/" + sub);

  constexpr int kEtaBins = 40;
  constexpr double kEtaMax = 3.2;
  constexpr int kPtBins = 50;
  constexpr double kPtMax = 200.;
  constexpr int kEBins = 50;
  constexpr double kEMax = 500.;

  p.denomEta = ib.book1D("denom_eta", "Selected truth objects vs #eta;#eta;objects", kEtaBins, -kEtaMax, kEtaMax);
  p.denomPt = ib.book1D("denom_pt", "Selected truth objects vs p_{T};p_{T} [GeV];objects", kPtBins, 0., kPtMax);
  p.denomEnergy = ib.book1D("denom_energy", "Selected truth objects vs E;E [GeV];objects", kEBins, 0., kEMax);
  p.effNumEta =
      ib.book1D("effnum_eta", "Branch-reproduced truth objects vs #eta;#eta;objects", kEtaBins, -kEtaMax, kEtaMax);
  p.effNumPt =
      ib.book1D("effnum_pt", "Branch-reproduced truth objects vs p_{T};p_{T} [GeV];objects", kPtBins, 0., kPtMax);
  p.effNumEnergy =
      ib.book1D("effnum_energy", "Branch-reproduced truth objects vs E;E [GeV];objects", kEBins, 0., kEMax);

  p.purity = ib.book1D("purity", "Branch hit purity;purity;objects", 52, -0.01, 1.03);
  p.completenessHits = ib.book1D("completeness_hits", "Branch hit completeness;completeness;objects", 52, -0.01, 1.03);
  p.completenessEnergy =
      ib.book1D("completeness_energy", "Branch energy completeness;completeness;objects", 52, -0.01, 1.03);
  p.energyResponse =
      ib.book1D("energy_response", "Branch sim-energy containment;E^{sim}_{Branch}/E_{gen};objects", 60, 0., 1.5);
  // Deposited-scale response is a closure test: == 1 by construction (the object's
  // per-cell fraction is its tracks' share of the deposit, i.e. the Branch's own
  // sim energy on that cell), so any deviation flags a fraction/deposit bug in PR
  // validation. Reconstructed-scale response is the informative one.
  p.rawEnergyResponseSim =
      ib.book1D("raw_energy_response_sim",
                "Branch raw energy response (deposited, closure: ==1);E^{sim}_{Branch}/E^{sim}_{hits};objects",
                80,
                0.,
                2.);
  p.rawEnergyResponseReco =
      ib.book1D("raw_energy_response_reco",
                "Branch raw energy response (reconstructed);E^{rec}_{Branch}/E^{rec}_{hits};objects",
                80,
                0.,
                4.);

  p.purityVsEta =
      ib.bookProfile("purity_vs_eta", "Branch hit purity vs #eta;#eta;purity", kEtaBins, -kEtaMax, kEtaMax, 0., 1.05);
  p.completenessVsEta = ib.bookProfile("completeness_vs_eta",
                                       "Branch energy completeness vs #eta;#eta;completeness",
                                       kEtaBins,
                                       -kEtaMax,
                                       kEtaMax,
                                       0.,
                                       1.05);
  p.responseVsEta = ib.bookProfile("response_vs_eta",
                                   "Branch sim-energy containment vs #eta;#eta;E^{sim}_{Branch}/E_{gen}",
                                   kEtaBins,
                                   -kEtaMax,
                                   kEtaMax,
                                   0.,
                                   1.5);
  p.responseVsEnergy = ib.bookProfile("response_vs_energy",
                                      "Branch sim-energy containment vs E;E [GeV];E^{sim}_{Branch}/E_{gen}",
                                      kEBins,
                                      0.,
                                      kEMax,
                                      0.,
                                      1.5);
  p.rawResponseSimVsEnergy = ib.bookProfile(
      "raw_response_sim_vs_energy",
      "Branch raw energy response (deposited, closure: ==1) vs E;E [GeV];E^{sim}_{Branch}/E^{sim}_{hits}",
      kEBins,
      0.,
      kEMax,
      0.,
      2.);
  p.rawResponseRecoVsEnergy =
      ib.bookProfile("raw_response_reco_vs_energy",
                     "Branch raw energy response (reconstructed) vs E;E [GeV];E^{rec}_{Branch}/E^{rec}_{hits}",
                     kEBins,
                     0.,
                     kEMax,
                     0.,
                     4.);

  // Best hit-matched Branch per object (the "other way around" view).
  p.bestPurity = ib.book1D("bestmatch_purity", "Best-match Branch hit purity;purity;objects", 52, -0.01, 1.03);
  p.bestCompletenessHits = ib.book1D(
      "bestmatch_completeness_hits", "Best-match Branch hit completeness;completeness;objects", 52, -0.01, 1.03);
  p.bestCompletenessEnergy = ib.book1D(
      "bestmatch_completeness_energy", "Best-match Branch energy completeness;completeness;objects", 52, -0.01, 1.03);
  p.bestResponse = ib.book1D(
      "bestmatch_response", "Best-match Branch sim-energy containment;E^{sim}_{Branch}/E_{gen};objects", 60, 0., 1.5);
  p.selfMatchEta = ib.book1D(
      "selfmatch_eta", "Objects whose best Branch is the natural one vs #eta;#eta;objects", kEtaBins, -kEtaMax, kEtaMax);
  p.selfMatchPt = ib.book1D(
      "selfmatch_pt", "Objects whose best Branch is the natural one vs p_{T};p_{T} [GeV];objects", kPtBins, 0., kPtMax);
  p.nSharingBranches = ib.book1D(
      "n_sharing_branches", "Distinct Branches sharing >=10% of the object hits;#Branches;objects", 51, -0.5, 50.5);
}

void BranchHGCalValidator::bookHistograms(DQMStore::IBooker& ib, edm::Run const&, edm::EventSetup const&) {
  book(ib, caloParticlePlots_, "CaloParticle");
  book(ib, simClusterPlots_, "SimCluster");
}

namespace {
  // logical-particle id <- SimTrack trackId, via the raw-graph node back-reference.
  std::unordered_map<uint32_t, uint32_t> buildTrackIdToParticle(truth::Graph const& graph, TruthGraph const& raw) {
    std::unordered_map<uint32_t, uint32_t> out;
    out.reserve(graph.nParticles());
    for (uint32_t i = 0; i < graph.nParticles(); ++i) {
      const int32_t simNode = graph.particles()[i].simNode;
      if (simNode < 0 || static_cast<uint32_t>(simNode) >= raw.nNodes())
        continue;
      auto const& nr = raw.nodeRef(static_cast<uint32_t>(simNode));
      if (nr.kind == TruthGraph::NodeKind::SimTrack)
        out[static_cast<uint32_t>(nr.key)] = i;
    }
    return out;
  }
}  // namespace

template <class Collection>
void BranchHGCalValidator::validate(Collection const& objects,
                                    truth::Graph const& graph,
                                    TruthGraph const& raw,
                                    truth::LogicalGraphHitIndex const& hitIndex,
                                    truth::BranchHitAssociator const& assoc,
                                    std::unordered_map<uint32_t, uint32_t> const& tidToParticle,
                                    std::unordered_map<uint32_t, float> const& cellSimEnergy,
                                    std::unordered_map<uint32_t, float> const& recHitEnergyByDetId,
                                    Plots& plots) {
  auto recoEnergyOf = [&recHitEnergyByDetId](uint32_t detId) -> double {
    auto it = recHitEnergyByDetId.find(detId);
    return it != recHitEnergyByDetId.end() ? static_cast<double>(it->second) : 0.;
  };
  for (auto const& obj : objects) {
    if (obj.g4Tracks().empty())
      continue;

    const double eta = obj.eta();
    const double pt = obj.pt();
    const double energy = obj.energy();
    if (pt < minPt_ || std::abs(eta) > maxEta_)
      continue;

    auto const& hitsAndFractions = obj.hits_and_fractions();
    if (hitsAndFractions.empty())
      continue;

    // Selected object: fills the efficiency denominator.
    plots.denomEta->Fill(eta);
    plots.denomPt->Fill(pt);
    plots.denomEnergy->Fill(energy);

    const uint32_t trackId = obj.g4Tracks().front().trackId();
    auto it = tidToParticle.find(trackId);
    if (it == tidToParticle.end())
      continue;  // unmapped -> counts as inefficiency
    const uint32_t particleId = it->second;

    // Branch subgraph calo hits for the mapped logical particle. branchEnergy is the
    // total deposited (sim) energy; branchCellEnergy is its per-cell breakdown, used
    // to restrict the raw response to the object's own footprint (a tiny object whose
    // trackId maps to a large shower would otherwise blow up the un-thresholded sim
    // ratio - the reco ratio stays finite only because the extra cells lack RecHits).
    std::unordered_map<uint32_t, double> branchCellEnergy;
    double branchEnergy = 0.;
    for (auto const& hit : hitIndex.subgraphHits(truth::HitChannel::HGCalCalo, particleId)) {
      branchCellEnergy[hit.detId] += hit.energy;
      branchEnergy += hit.energy;
    }

    std::vector<truth::RecoHit> recoHits;
    recoHits.reserve(hitsAndFractions.size());
    uint32_t shared = 0;
    double totalFraction = 0.;
    double sharedFraction = 0.;
    // Raw energy response references, all on the *object's* cells: the object's own
    // deposited (sim) and reconstructed (rec) energy -- fraction-weighted, the standard
    // CaloParticle/SimCluster convention -- and the Branch's energy on those same cells
    // (its per-cell sim deposit; the whole-cell RecHit it claims). The sim and reco
    // responses then differ only by the deposited-vs-reconstructed scale.
    double objectSimEnergy = 0.;
    double objectRecoEnergy = 0.;
    double branchSimOnObject = 0.;
    double branchRecoOnObject = 0.;
    for (auto const& [detId, fraction] : hitsAndFractions) {
      recoHits.push_back(truth::RecoHit{detId, 1.f, fraction});
      totalFraction += fraction;
      if (auto cs = cellSimEnergy.find(detId); cs != cellSimEnergy.end())
        objectSimEnergy += static_cast<double>(fraction) * static_cast<double>(cs->second);
      objectRecoEnergy += static_cast<double>(fraction) * recoEnergyOf(detId);
      if (auto be = branchCellEnergy.find(detId); be != branchCellEnergy.end()) {
        ++shared;
        sharedFraction += fraction;
        branchSimOnObject += be->second;
        branchRecoOnObject += recoEnergyOf(detId);
      }
    }

    const double completenessHits = static_cast<double>(shared) / hitsAndFractions.size();
    const double purity = branchCellEnergy.empty() ? 0. : static_cast<double>(shared) / branchCellEnergy.size();
    const double completenessEnergy = totalFraction > 0. ? sharedFraction / totalFraction : 0.;
    // Energy containment: Branch subgraph sim-hit energy over the object energy.
    // (CaloParticle::simEnergy() is not populated in these samples, so the
    // gen-level energy is the reference; the ratio reflects the active-material
    // sampling fraction and so varies by detector region.)
    const double response = energy > 0. ? branchEnergy / energy : 0.;

    plots.purity->Fill(purity);
    plots.completenessHits->Fill(completenessHits);
    plots.completenessEnergy->Fill(completenessEnergy);
    plots.purityVsEta->Fill(eta, purity);
    plots.completenessVsEta->Fill(eta, completenessEnergy);
    if (energy > 0.) {
      plots.energyResponse->Fill(response);
      plots.responseVsEta->Fill(eta, response);
      plots.responseVsEnergy->Fill(energy, response);
    }

    // Raw energy response: the Branch's energy on the object's footprint normalised
    // by the object's own hit energy (rather than its generator energy), on the
    // deposited and reconstructed scales. The deposited ratio is == 1 by construction
    // (closure / PR-validation invariant); the reconstructed ratio is informative.
    if (objectSimEnergy > 0.) {
      const double rawSim = branchSimOnObject / objectSimEnergy;
      plots.rawEnergyResponseSim->Fill(rawSim);
      plots.rawResponseSimVsEnergy->Fill(energy, rawSim);
    }
    if (objectRecoEnergy > 0.) {
      const double rawReco = branchRecoOnObject / objectRecoEnergy;
      plots.rawEnergyResponseReco->Fill(rawReco);
      plots.rawResponseRecoVsEnergy->Fill(energy, rawReco);
    }

    // Reproduction efficiency numerator: the associator picks this particle's
    // branch as the best (tightest among equally-best-scoring) hit match.
    auto matches = assoc.bestBranches(std::span<const truth::RecoHit>(recoHits));
    if (!matches.empty()) {
      const float bestScore = matches.front().score;
      uint32_t tightest = matches.front().rootParticleId;
      std::size_t tightestSize = hitIndex.subgraphHits(truth::HitChannel::HGCalCalo, tightest).size();
      for (auto const& m : matches) {
        if (m.score > bestScore)
          break;
        const std::size_t size = hitIndex.subgraphHits(truth::HitChannel::HGCalCalo, m.rootParticleId).size();
        if (size < tightestSize) {
          tightestSize = size;
          tightest = m.rootParticleId;
        }
      }
      if (tightest == particleId) {
        plots.effNumEta->Fill(eta);
        plots.effNumPt->Fill(pt);
        plots.effNumEnergy->Fill(energy);
      }

      // --- "Other way around": the best hit-matched Branch's own performance. ---
      // Self-match: the best Branch is the natural (trackId-seeded) one.
      if (tightest == particleId) {
        plots.selfMatchEta->Fill(eta);
        plots.selfMatchPt->Fill(pt);
      }

      // Merge/split: how many distinct Branches share >=10% of the object's hits.
      // For the SharedHits metric, BranchMatch::sharedEnergy is the shared-cell count.
      const double shareThreshold = 0.1 * static_cast<double>(hitsAndFractions.size());
      uint32_t nSharing = 0;
      for (auto const& m : matches)
        if (static_cast<double>(m.sharedEnergy) >= shareThreshold)
          ++nSharing;
      plots.nSharingBranches->Fill(std::min<uint32_t>(nSharing, 50));

      // Best Branch's purity / completeness / response vs the object.
      std::unordered_set<uint32_t> bestDetIds;
      double bestBranchEnergy = 0.;
      for (auto const& hit : hitIndex.subgraphHits(truth::HitChannel::HGCalCalo, tightest)) {
        bestDetIds.insert(hit.detId);
        bestBranchEnergy += hit.energy;
      }
      uint32_t bestShared = 0;
      double bestSharedFraction = 0.;
      for (auto const& [detId, fraction] : hitsAndFractions) {
        if (bestDetIds.count(detId)) {
          ++bestShared;
          bestSharedFraction += fraction;
        }
      }
      plots.bestPurity->Fill(bestDetIds.empty() ? 0. : static_cast<double>(bestShared) / bestDetIds.size());
      plots.bestCompletenessHits->Fill(static_cast<double>(bestShared) / hitsAndFractions.size());
      plots.bestCompletenessEnergy->Fill(totalFraction > 0. ? bestSharedFraction / totalFraction : 0.);
      if (energy > 0.)
        plots.bestResponse->Fill(bestBranchEnergy / energy);
    }
  }
}

std::unordered_map<uint32_t, float> BranchHGCalValidator::collectRecHitEnergyByDetId(edm::Event const& event) const {
  std::unordered_map<uint32_t, float> energies;
  for (uint32_t i = 0; i < hgcalRecHitTokens_.size(); ++i) {
    edm::Handle<HGCRecHitCollection> handle;
    event.getByToken(hgcalRecHitTokens_[i], handle);
    if (!handle.isValid()) {
      edm::LogWarning("BranchHGCalValidator")
          << "Missing HGCRecHit collection " << hgcalRecHitTags_[i].encode() << "; skipping it.";
      continue;
    }
    energies.reserve(energies.size() + handle->size());
    for (auto const& hit : *handle)
      energies.emplace(hit.detid().rawId(), hit.energy());  // keep first for duplicate DetIds
  }
  for (uint32_t i = 0; i < pfRecHitTokens_.size(); ++i) {
    edm::Handle<reco::PFRecHitCollection> handle;
    event.getByToken(pfRecHitTokens_[i], handle);
    if (!handle.isValid()) {
      edm::LogWarning("BranchHGCalValidator")
          << "Missing reco::PFRecHitCollection " << pfRecHitTags_[i].encode() << "; skipping it.";
      continue;
    }
    energies.reserve(energies.size() + handle->size());
    for (auto const& hit : *handle)
      energies.emplace(hit.detId(), hit.energy());
  }
  return energies;
}

void BranchHGCalValidator::analyze(edm::Event const& event, edm::EventSetup const&) {
  auto const& graph = event.get(graphToken_);
  auto const& raw = event.get(rawToken_);
  auto const& hitIndex = event.get(hitIndexToken_);

  const auto tidToParticle = buildTrackIdToParticle(graph, raw);
  truth::BranchHitAssociator assoc(hitIndex, {}, truth::BranchHitAssociator::Metric::SharedHits);

  // Per-cell deposited (sim) energy = sum of every particle's direct HGCalCalo hits
  // in that cell (each PCaloHit belongs to exactly one SimTrack), and per-cell
  // reconstructed energy from the RecHit collections; both keyed by DetId so the
  // raw response can normalise a Branch's energy by the object's own hit energy.
  std::unordered_map<uint32_t, float> cellSimEnergy;
  for (uint32_t p = 0; p < hitIndex.nParticles(); ++p)
    for (auto const& hit : hitIndex.directHits(truth::HitChannel::HGCalCalo, p))
      cellSimEnergy[hit.detId] += hit.energy;
  const auto recHitEnergyByDetId = collectRecHitEnergyByDetId(event);

  validate(event.get(caloParticleToken_),
           graph,
           raw,
           hitIndex,
           assoc,
           tidToParticle,
           cellSimEnergy,
           recHitEnergyByDetId,
           caloParticlePlots_);
  validate(event.get(simClusterToken_),
           graph,
           raw,
           hitIndex,
           assoc,
           tidToParticle,
           cellSimEnergy,
           recHitEnergyByDetId,
           simClusterPlots_);
}

void BranchHGCalValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("rawSrc", edm::InputTag("truthGraphProducer"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<edm::InputTag>("caloParticles", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("simClusters", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<std::string>("folder", "HGCAL/BranchValidator");
  desc.add<double>("minPt", 1.0);
  desc.add<double>("maxEta", 3.0);
  // RecHit collections for the raw (reconstructed) energy response, in the same
  // order DetIdToRecHitMapProducer used to build the DetId->RecHit map.
  desc.add<std::vector<edm::InputTag>>("hgcalRecHits",
                                       {edm::InputTag("HGCalRecHit", "HGCEERecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEFRecHits"),
                                        edm::InputTag("HGCalRecHit", "HGCHEBRecHits")});
  desc.add<std::vector<edm::InputTag>>("pfRecHits",
                                       {edm::InputTag("particleFlowRecHitECAL", "Cleaned"),
                                        edm::InputTag("particleFlowRecHitHBHE", "Cleaned"),
                                        edm::InputTag("particleFlowRecHitHF", "Cleaned"),
                                        edm::InputTag("particleFlowRecHitHO", "Cleaned")});
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(BranchHGCalValidator);
