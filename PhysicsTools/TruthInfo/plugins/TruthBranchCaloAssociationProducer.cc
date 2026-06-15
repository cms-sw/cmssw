// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

// Builds TICL-style AssociationMaps between the truth::Branch graph and the legacy
// HGCAL calo truth objects (CaloParticle, SimCluster), in the same fashion as the
// TICL trackster<->simTrackster associators. For each direction it stores, per
// object, the matched branches with their shared energy and score (lower = better),
// sorted so the best-matched branch is first. The branch side is restricted to the
// "interesting" particles configured via interestingPdgIds (empty = all), so the
// association metrics are computed against the particles of interest. Downstream
// DQM validators consume these maps to compute efficiency / fake / merge /
// duplicate / purity, exactly like HGCalValidator consumes its association maps.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"

#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"
#include "PhysicsTools/TruthInfo/interface/Graph.h"
#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndex.h"

namespace {
  // Raw-index AssociationMap (object index -> [(branch id, sharedEnergy, score)]).
  using BranchAssociationMap = ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore>;
}  // namespace

class TruthBranchCaloAssociationProducer : public edm::stream::EDProducer<> {
public:
  explicit TruthBranchCaloAssociationProducer(edm::ParameterSet const&);
  void produce(edm::Event&, edm::EventSetup const&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  template <class Collection>
  void associate(Collection const& objects,
                 truth::Graph const& graph,
                 truth::BranchHitAssociator const& assoc,
                 std::string const& recoToBranchLabel,
                 std::string const& branchToRecoLabel,
                 edm::Event& event) const;

  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<CaloParticle>> caloParticleToken_;
  const edm::EDGetTokenT<std::vector<SimCluster>> simClusterToken_;
  const std::vector<int> interestingPdgIds_;
};

TruthBranchCaloAssociationProducer::TruthBranchCaloAssociationProducer(edm::ParameterSet const& cfg)
    : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
      hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
      caloParticleToken_(consumes<std::vector<CaloParticle>>(cfg.getParameter<edm::InputTag>("caloParticles"))),
      simClusterToken_(consumes<std::vector<SimCluster>>(cfg.getParameter<edm::InputTag>("simClusters"))),
      interestingPdgIds_(cfg.getParameter<std::vector<int>>("interestingPdgIds")) {
  produces<BranchAssociationMap>("caloParticleToBranch");
  produces<BranchAssociationMap>("branchToCaloParticle");
  produces<BranchAssociationMap>("simClusterToBranch");
  produces<BranchAssociationMap>("branchToSimCluster");
}

template <class Collection>
void TruthBranchCaloAssociationProducer::associate(Collection const& objects,
                                                   truth::Graph const& graph,
                                                   truth::BranchHitAssociator const& assoc,
                                                   std::string const& recoToBranchLabel,
                                                   std::string const& branchToRecoLabel,
                                                   edm::Event& event) const {
  auto recoToBranch = std::make_unique<BranchAssociationMap>(static_cast<unsigned int>(objects.size()));
  auto branchToReco = std::make_unique<BranchAssociationMap>(graph.nParticles());

  std::vector<truth::RecoHit> recoHits;
  unsigned int objIndex = 0;
  for (auto const& obj : objects) {
    recoHits.clear();
    for (auto const& [detId, fraction] : obj.hits_and_fractions())
      recoHits.push_back(truth::RecoHit{detId, 1.f, fraction});

    if (!recoHits.empty()) {
      for (auto const& m : assoc.bestBranches(std::span<const truth::RecoHit>(recoHits))) {
        recoToBranch->insert(objIndex, m.rootParticleId, m.sharedEnergy, m.score);
        branchToReco->insert(m.rootParticleId, objIndex, m.sharedEnergy, m.score);
      }
    }
    ++objIndex;
  }

  // Sort each row by score (ascending) so the best-matched branch/object is first.
  recoToBranch->sort(true);
  branchToReco->sort(true);

  event.put(std::move(recoToBranch), recoToBranchLabel);
  event.put(std::move(branchToReco), branchToRecoLabel);
}

void TruthBranchCaloAssociationProducer::produce(edm::Event& event, edm::EventSetup const&) {
  auto const& graph = event.get(graphToken_);
  auto const& hitIndex = event.get(hitIndexToken_);

  // Candidate branch roots = the interesting particles (empty config -> all).
  std::vector<uint32_t> roots;
  if (!interestingPdgIds_.empty()) {
    for (uint32_t i = 0; i < graph.nParticles(); ++i) {
      const int pdgId = graph.particles[i].pdgId;
      if (std::find(interestingPdgIds_.begin(), interestingPdgIds_.end(), pdgId) != interestingPdgIds_.end())
        roots.push_back(i);
    }
  }

  // SharedEnergy metric -> score is the normalized shared-energy penalty (lower is
  // better), matching the TICL association-score convention.
  truth::BranchHitAssociator assoc(hitIndex, roots, truth::BranchHitAssociator::Metric::SharedEnergy);

  associate(event.get(caloParticleToken_), graph, assoc, "caloParticleToBranch", "branchToCaloParticle", event);
  associate(event.get(simClusterToken_), graph, assoc, "simClusterToBranch", "branchToSimCluster", event);
}

void TruthBranchCaloAssociationProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<edm::InputTag>("caloParticles", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("simClusters", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<std::vector<int>>("interestingPdgIds", {})
      ->setComment("Restrict the branch side to these PDG ids (the interesting particles); empty = all branches.");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(TruthBranchCaloAssociationProducer);
