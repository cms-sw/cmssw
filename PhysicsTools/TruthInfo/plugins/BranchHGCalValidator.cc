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
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"

#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"
#include "PhysicsTools/TruthInfo/interface/Graph.h"
#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "PhysicsTools/TruthInfo/interface/TruthGraph.h"

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
    // Profiles vs kinematics.
    MonitorElement* purityVsEta = nullptr;
    MonitorElement* completenessVsEta = nullptr;
    MonitorElement* responseVsEta = nullptr;
    MonitorElement* responseVsEnergy = nullptr;
  };

  void book(DQMStore::IBooker&, Plots&, std::string const& sub);

  template <class Collection>
  void validate(Collection const& objects,
                truth::Graph const& graph,
                TruthGraph const& raw,
                truth::LogicalGraphHitIndex const& hitIndex,
                truth::BranchHitAssociator const& assoc,
                std::unordered_map<uint32_t, uint32_t> const& tidToParticle,
                Plots& plots);

  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<TruthGraph> rawToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<CaloParticle>> caloParticleToken_;
  const edm::EDGetTokenT<std::vector<SimCluster>> simClusterToken_;

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
      maxEta_(cfg.getParameter<double>("maxEta")) {}

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
      const int32_t simNode = graph.particles[i].simNode;
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
                                    Plots& plots) {
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

    // Branch subgraph calo hits for the mapped logical particle.
    std::unordered_set<uint32_t> branchDetIds;
    double branchEnergy = 0.;
    for (auto const& hit : hitIndex.subgraphHits(truth::HitChannel::HGCalCalo, particleId)) {
      branchDetIds.insert(hit.detId);
      branchEnergy += hit.energy;
    }

    std::vector<truth::RecoHit> recoHits;
    recoHits.reserve(hitsAndFractions.size());
    uint32_t shared = 0;
    double totalFraction = 0.;
    double sharedFraction = 0.;
    for (auto const& [detId, fraction] : hitsAndFractions) {
      recoHits.push_back(truth::RecoHit{detId, 1.f, fraction});
      totalFraction += fraction;
      if (branchDetIds.count(detId)) {
        ++shared;
        sharedFraction += fraction;
      }
    }

    const double completenessHits = static_cast<double>(shared) / hitsAndFractions.size();
    const double purity = branchDetIds.empty() ? 0. : static_cast<double>(shared) / branchDetIds.size();
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
    }
  }
}

void BranchHGCalValidator::analyze(edm::Event const& event, edm::EventSetup const&) {
  auto const& graph = event.get(graphToken_);
  auto const& raw = event.get(rawToken_);
  auto const& hitIndex = event.get(hitIndexToken_);

  const auto tidToParticle = buildTrackIdToParticle(graph, raw);
  truth::BranchHitAssociator assoc(hitIndex, {}, truth::BranchHitAssociator::Metric::SharedHits);

  validate(event.get(caloParticleToken_), graph, raw, hitIndex, assoc, tidToParticle, caloParticlePlots_);
  validate(event.get(simClusterToken_), graph, raw, hitIndex, assoc, tidToParticle, simClusterPlots_);
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
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(BranchHGCalValidator);
