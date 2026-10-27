// Flat nanoAOD tables of the MC-truth graph as a training truth.
//
// TruthParticleFlatTableProducer writes one row per truth particle of a target list, the
// list a TruthBranchTargetsProducer publishes for one graph level (the caloBoundary level
// by default: one particle per outward tracker-to-calorimeter crossing, the SimCluster
// analogue), optionally only those with sim energy in HGCAL, and publishes the ids it
// wrote. TruthBranchAssociationFlatTableProducer writes one reco-to-branch association
// map of SimGeneral/TruthGraphAssociatorProducers in the count/offset + flat link layout
// of the TICL association tables, with the branch key remapped from the graph particle id
// to the row of that particle table (it reads the published ids), so a consumer indexes
// the two tables directly.
//
// Both read the map and the graph in compiled code: the association-map template cannot
// be instantiated from FWLite or cppyy.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/InteractionId.h"
#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"

namespace {

  // The detectors simEnergyHGCAL sums over.
  bool isHGCal(uint32_t rawDetId) {
    const auto det = DetId(rawDetId).det();
    return det == DetId::HGCalEE || det == DetId::HGCalHSi || det == DetId::HGCalHSc;
  }

  // The first particle that produced `id`, or -1 for a root. The first production vertex
  // and its first incoming particle other than `id` itself; a GEN particle has one.
  int32_t firstParent(truth::Graph const& graph, uint32_t id) {
    for (const uint32_t vertexId : graph.productionVertices(id)) {
      if (vertexId >= graph.nVertices()) {
        continue;
      }
      for (const uint32_t parent : graph.incomingParticles(vertexId)) {
        if (parent != id && parent < graph.nParticles()) {
          return static_cast<int32_t>(parent);
        }
      }
    }
    return -1;
  }

  // The topmost GEN ancestor along the first-parent chain, or -1 when neither the
  // particle nor any ancestor has a GEN record. A gun particle is its own GEN root, a
  // conversion electron's is the photon, a tau decay product's is the tau or the gun
  // mother above it. Bounded by the particle count, so a malformed cyclic history ends.
  int32_t genRoot(truth::Graph const& graph, uint32_t id) {
    int32_t last = graph.particles()[id].hasGen() ? static_cast<int32_t>(id) : -1;
    uint32_t current = id;
    for (uint32_t step = 0; step < graph.nParticles(); ++step) {
      const int32_t parent = firstParent(graph, current);
      if (parent < 0) {
        break;
      }
      current = static_cast<uint32_t>(parent);
      if (graph.particles()[current].hasGen()) {
        last = parent;
      }
    }
    return last;
  }

  // The sorted target list, the key space of the particle table. A list is sorted as the
  // targets producer emits it; the copy only happens when a caller hands an unsorted one.
  std::vector<unsigned int> sortedTargets(std::vector<unsigned int> const& targets) {
    if (std::is_sorted(targets.begin(), targets.end())) {
      return targets;
    }
    std::vector<unsigned int> sorted = targets;
    std::sort(sorted.begin(), sorted.end());
    return sorted;
  }

}  // namespace

class TruthParticleFlatTableProducer : public edm::global::EDProducer<> {
public:
  explicit TruthParticleFlatTableProducer(edm::ParameterSet const& cfg)
      : graphToken_(consumes<truth::Graph>(cfg.getParameter<edm::InputTag>("src"))),
        hitIndexToken_(consumes<truth::LogicalGraphHitIndex>(cfg.getParameter<edm::InputTag>("hitIndex"))),
        targetsToken_(consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("targets"))),
        name_(cfg.getParameter<std::string>("name")),
        doc_(cfg.getParameter<std::string>("doc")),
        skipNonExistingSrc_(cfg.getParameter<bool>("skipNonExistingSrc")),
        requireHGCALHits_(cfg.getParameter<bool>("requireHGCALHits")),
        precision_(cfg.getParameter<int>("precision")) {
    produces<nanoaod::FlatTable>();
    // The graph ids of the rows written, in row order: the key space of the table, which
    // the association table producers read so their link indices are rows of THIS table.
    produces<std::vector<unsigned int>>("selected");
  }

  void produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<truth::Graph> graphToken_;
  const edm::EDGetTokenT<truth::LogicalGraphHitIndex> hitIndexToken_;
  const edm::EDGetTokenT<std::vector<unsigned int>> targetsToken_;
  const std::string name_;
  const std::string doc_;
  const bool skipNonExistingSrc_;
  const bool requireHGCALHits_;
  const int precision_;
};

void TruthParticleFlatTableProducer::produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const {
  auto const graphHandle = event.getHandle(graphToken_);
  auto const hitIndexHandle = event.getHandle(hitIndexToken_);
  auto const targetsHandle = event.getHandle(targetsToken_);
  const bool present = graphHandle.isValid() && hitIndexHandle.isValid() && targetsHandle.isValid();
  if (!present && !skipNonExistingSrc_) {
    throw cms::Exception("ProductNotFound") << "TruthParticleFlatTableProducer '" << name_
                                            << "': the truth graph, the hit index or the target list is missing";
  }

  std::vector<uint32_t> graphId, levelFlags;
  std::vector<int32_t> pdgId, bunchCrossing, eventIndex, parentGraphId, genRootGraphId;
  std::vector<uint8_t> isSignal, hasBoundary;
  std::vector<float> charge, pt, eta, phi, energy;
  std::vector<float> boundaryX, boundaryY, boundaryZ, boundaryT;
  std::vector<float> boundaryPx, boundaryPy, boundaryPz, boundaryE, boundaryPt, boundaryEta, boundaryPhi;
  std::vector<float> simEnergyHGCAL;

  if (present) {
    auto const& graph = *graphHandle;
    auto const& hitIndex = *hitIndexHandle;
    const std::vector<unsigned int> targets = sortedTargets(*targetsHandle);
    const std::size_t n = targets.size();
    for (auto* v : {&graphId, &levelFlags}) {
      v->reserve(n);
    }
    std::vector<truth::LogicalGraphHitIndex::Hit> hits;
    for (const unsigned int id : targets) {
      if (id >= graph.nParticles()) {
        throw cms::Exception("Configuration")
            << "TruthParticleFlatTableProducer '" << name_ << "': target particle " << id
            << " is outside the truth graph (" << graph.nParticles() << " particles). The target list and the "
            << "graph must come from the same event.";
      }
      auto const& data = graph.particles()[id];
      const truth::Particle particle = graph.particle(id);

      // The sim energy of the particle and of every Geant4 descendant in HGCAL: what a
      // SimCluster calls simEnergy. Each index entry is one (particle, cell) deposit in
      // either storage layout, so the plain sum is the total. With requireHGCALHits a
      // particle that left nothing in HGCAL (a barrel crosser, a pile-up particle
      // absorbed before the calorimeter) is not a row: no HGCAL reco object can share
      // energy with it, and at PU 30 that is three rows out of four.
      hits.clear();
      hitIndex.appendSubgraphHits(truth::HitChannel::Calo, id, hits);
      double simEnergy = 0.;
      for (auto const& hit : hits) {
        if (isHGCal(hit.detId)) {
          simEnergy += hit.energy;
        }
      }
      if (requireHGCALHits_ && simEnergy <= 0.) {
        continue;
      }
      simEnergyHGCAL.push_back(static_cast<float>(simEnergy));

      graphId.push_back(id);
      pdgId.push_back(data.pdgId);
      charge.push_back(static_cast<float>(particle.charge()));
      isSignal.push_back(data.isSignal() ? 1 : 0);
      bunchCrossing.push_back(data.bunchCrossing());
      eventIndex.push_back(data.eventIndex());
      levelFlags.push_back(data.levelFlags);
      parentGraphId.push_back(firstParent(graph, id));
      genRootGraphId.push_back(genRoot(graph, id));

      // A zero momentum has no direction: a pile-up GEN particle nothing gave a momentum.
      if (data.hasMomentum()) {
        pt.push_back(static_cast<float>(data.momentum.pt()));
        eta.push_back(static_cast<float>(data.momentum.eta()));
        phi.push_back(static_cast<float>(data.momentum.phi()));
        energy.push_back(static_cast<float>(data.momentum.energy()));
      } else {
        pt.push_back(0.f);
        eta.push_back(0.f);
        phi.push_back(0.f);
        energy.push_back(0.f);
      }

      // Checkpoint 0 is the outward crossing of the tracker-calorimeter boundary. Every
      // caloBoundary member has one; a target of another level may not, and then the
      // boundary columns hold zeros and hasBoundary says so.
      const auto boundary = particle.checkpoint(0);
      hasBoundary.push_back(boundary.has_value() ? 1 : 0);
      if (boundary) {
        auto const& pos = boundary->position;
        auto const& mom = boundary->momentum;
        boundaryX.push_back(pos.x());
        boundaryY.push_back(pos.y());
        boundaryZ.push_back(pos.z());
        boundaryT.push_back(pos.t());
        boundaryPx.push_back(mom.px());
        boundaryPy.push_back(mom.py());
        boundaryPz.push_back(mom.pz());
        boundaryE.push_back(mom.energy());
        const bool hasDirection = mom.px() != 0.f || mom.py() != 0.f || mom.pz() != 0.f;
        boundaryPt.push_back(hasDirection ? mom.pt() : 0.f);
        boundaryEta.push_back(hasDirection ? mom.eta() : 0.f);
        boundaryPhi.push_back(hasDirection ? mom.phi() : 0.f);
      } else {
        for (auto* v : {&boundaryX, &boundaryY, &boundaryZ, &boundaryT, &boundaryPx, &boundaryPy, &boundaryPz,
                        &boundaryE, &boundaryPt, &boundaryEta, &boundaryPhi}) {
          v->push_back(0.f);
        }
      }
    }
  }

  auto table = std::make_unique<nanoaod::FlatTable>(graphId.size(), name_, false, false);
  table->setDoc(doc_);
  table->addColumn<uint32_t>("graphId", graphId, "Particle id in the truth graph (key of the association maps)");
  table->addColumn<int32_t>("pdgId", pdgId, "PDG id");
  table->addColumn<float>("charge", charge, "Electric charge [e]", precision_);
  table->addColumn<bool>("isSignal", isSignal, "True for the signal interaction, false for pile-up");
  table->addColumn<int32_t>("bunchCrossing", bunchCrossing, "Bunch crossing of the interaction (0 = in time)");
  table->addColumn<int32_t>("eventIndex", eventIndex, "Interaction index inside the bunch crossing (0 = signal)");
  table->addColumn<float>("pt", pt, "pT of the GEN momentum, or of the SimTrack for a SIM-only particle [GeV]", precision_);
  table->addColumn<float>("eta", eta, "eta of the GEN/SimTrack momentum", precision_);
  table->addColumn<float>("phi", phi, "phi of the GEN/SimTrack momentum", precision_);
  table->addColumn<float>("energy", energy, "Energy of the GEN/SimTrack momentum [GeV]", precision_);
  table->addColumn<bool>("hasBoundary", hasBoundary, "Checkpoint at the tracker-calorimeter boundary recorded");
  table->addColumn<float>("boundaryX", boundaryX, "x at the calo boundary [cm]", precision_);
  table->addColumn<float>("boundaryY", boundaryY, "y at the calo boundary [cm]", precision_);
  table->addColumn<float>("boundaryZ", boundaryZ, "z at the calo boundary [cm]", precision_);
  table->addColumn<float>("boundaryT", boundaryT, "t at the calo boundary [ns]", precision_);
  table->addColumn<float>("boundaryPx", boundaryPx, "px at the calo boundary [GeV]", precision_);
  table->addColumn<float>("boundaryPy", boundaryPy, "py at the calo boundary [GeV]", precision_);
  table->addColumn<float>("boundaryPz", boundaryPz, "pz at the calo boundary [GeV]", precision_);
  table->addColumn<float>("boundaryE", boundaryE, "Energy at the calo boundary [GeV]", precision_);
  table->addColumn<float>("boundaryPt", boundaryPt, "pT at the calo boundary [GeV]", precision_);
  table->addColumn<float>("boundaryEta", boundaryEta, "eta of the momentum at the calo boundary", precision_);
  table->addColumn<float>("boundaryPhi", boundaryPhi, "phi of the momentum at the calo boundary", precision_);
  table->addColumn<float>(
      "simEnergyHGCAL", simEnergyHGCAL, "Sim-hit energy of the particle and its descendants in HGCAL [GeV]", precision_);
  table->addColumn<uint32_t>("levelFlags", levelFlags, "Truth levels the particle belongs to (truth::LevelFlag bits)");
  table->addColumn<int32_t>("parentGraphId", parentGraphId, "Graph id of the first parent, -1 for a root");
  table->addColumn<int32_t>(
      "genRootGraphId", genRootGraphId, "Graph id of the topmost GEN ancestor (itself for a GEN root), -1 if none");
  event.put(std::move(table));
  event.put(std::make_unique<std::vector<unsigned int>>(graphId.begin(), graphId.end()), "selected");
}

void TruthParticleFlatTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
  desc.add<edm::InputTag>("hitIndex", edm::InputTag("truthLogicalGraphHitIndexProducer"));
  desc.add<edm::InputTag>("targets", edm::InputTag("truthBranchTargets", "truthToRecoTargetsCaloBoundary"))
      ->setComment("The particle ids to write, one row each, in ascending order; a level list of TruthBranchTargetsProducer");
  desc.add<std::string>("name", "TruthCB");
  desc.add<std::string>("doc", "MC-truth graph particles at the calo boundary");
  desc.add<bool>("skipNonExistingSrc", true)->setComment("Write an empty table when an input is missing");
  desc.add<bool>("requireHGCALHits", true)
      ->setComment("Write only the targets with sim energy in HGCAL; the others can match no HGCAL reco object");
  desc.add<int>("precision", -1);
  descriptions.add("truthParticleFlatTable", desc);
}

class TruthBranchAssociationFlatTableProducer : public edm::global::EDProducer<> {
public:
  using MapType = ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore>;

  explicit TruthBranchAssociationFlatTableProducer(edm::ParameterSet const& cfg)
      : mapToken_(consumes<MapType>(cfg.getParameter<edm::InputTag>("src"))),
        targetsToken_(consumes<std::vector<unsigned int>>(cfg.getParameter<edm::InputTag>("targets"))),
        name_(cfg.getParameter<std::string>("name")),
        linkName_(name_ + "Links"),
        doc_(cfg.getParameter<std::string>("doc")),
        recoToTruth_(cfg.getParameter<std::string>("direction") == "recoToTruth"),
        skipNonExistingSrc_(cfg.getParameter<bool>("skipNonExistingSrc")),
        precision_(cfg.getParameter<int>("precision")) {
    const auto direction = cfg.getParameter<std::string>("direction");
    if (direction != "recoToTruth" && direction != "truthToReco") {
      throw cms::Exception("Configuration")
          << "direction must be 'recoToTruth' or 'truthToReco', got '" << direction << "'";
    }
    // The main table, rows parallel to the reco collection (recoToTruth) or to the truth
    // particle table (truthToReco), and the flat link table, as the TICL association
    // tables lay them out: <name>_n<name>Links / <name>_o<name>Links index <name>Links.
    produces<nanoaod::FlatTable>();
    produces<nanoaod::FlatTable>(linkName_ + "Table");
  }

  void produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<MapType> mapToken_;
  const edm::EDGetTokenT<std::vector<unsigned int>> targetsToken_;
  const std::string name_;
  const std::string linkName_;
  const std::string doc_;
  const bool recoToTruth_;
  const bool skipNonExistingSrc_;
  const int precision_;
};

void TruthBranchAssociationFlatTableProducer::produce(edm::StreamID,
                                                      edm::Event& event,
                                                      edm::EventSetup const&) const {
  auto const mapHandle = event.getHandle(mapToken_);
  auto const targetsHandle = event.getHandle(targetsToken_);
  const bool present = mapHandle.isValid() && targetsHandle.isValid();
  if (!present && !skipNonExistingSrc_) {
    throw cms::Exception("ProductNotFound")
        << "TruthBranchAssociationFlatTableProducer '" << name_ << "': the association map or the target list is missing";
  }

  std::vector<int32_t> counts, offsets;
  std::vector<uint32_t> index, graphId;
  std::vector<float> value, score;

  if (present) {
    auto const& map = *mapHandle;
    const std::vector<unsigned int> targets = sortedTargets(*targetsHandle);
    // The row of a graph particle in the particle table, or -1 when it is not a target.
    auto rowOf = [&targets](unsigned int particleId) -> int64_t {
      const auto it = std::lower_bound(targets.begin(), targets.end(), particleId);
      if (it == targets.end() || *it != particleId) {
        return -1;
      }
      return it - targets.begin();
    };

    if (recoToTruth_) {
      // One row per reco object; each link points at a truth row. A candidate that is not
      // a target, which happens when the associator was given a wider candidate list than
      // the table, has no row to point at and is left out.
      counts.reserve(map.size());
      for (unsigned int reco = 0; reco < map.size(); ++reco) {
        int32_t count = 0;
        for (auto const& element : map[reco]) {
          const int64_t row = rowOf(element.index());
          if (row < 0) {
            continue;
          }
          index.push_back(static_cast<uint32_t>(row));
          graphId.push_back(element.index());
          value.push_back(element.sharedEnergy());
          score.push_back(element.score());
          ++count;
        }
        counts.push_back(count);
      }
    } else {
      // One row per target, in the order of the particle table; each link points at a reco
      // object. The map has a row per graph particle, so a target beyond it is an empty row.
      counts.reserve(targets.size());
      for (const unsigned int particleId : targets) {
        int32_t count = 0;
        if (particleId < map.size()) {
          for (auto const& element : map[particleId]) {
            index.push_back(element.index());
            graphId.push_back(particleId);
            value.push_back(element.sharedEnergy());
            score.push_back(element.score());
            ++count;
          }
        }
        counts.push_back(count);
      }
    }
    offsets.reserve(counts.size());
    int32_t offset = 0;
    for (const int32_t count : counts) {
      offsets.push_back(offset);
      offset += count;
    }
  }

  auto table = std::make_unique<nanoaod::FlatTable>(counts.size(), name_, false, false);
  table->setDoc(doc_);
  table->addColumn<int32_t>("n" + linkName_, counts, "Number of links of this row in " + linkName_);
  table->addColumn<int32_t>("o" + linkName_, offsets, "Offset of the first link of this row in " + linkName_);
  event.put(std::move(table));

  auto links = std::make_unique<nanoaod::FlatTable>(index.size(), linkName_, false, false);
  links->setDoc("Links of " + name_);
  if (recoToTruth_) {
    links->addColumn<uint32_t>("index", index, "Row in the truth particle table");
    links->addColumn<uint32_t>("graphId", graphId, "Truth graph particle id of the linked row");
    links->addColumn<float>(
        "sharedEnergy", value, "Energy the reco object shares with the truth branch [GeV]", precision_);
    links->addColumn<float>("score", score, "1 - reco purity: the reco energy the branch fails to cover (lower is better)", precision_);
  } else {
    links->addColumn<uint32_t>("index", index, "Row in the reco collection");
    links->addColumn<uint32_t>("graphId", graphId, "Truth graph particle id of this row");
    links->addColumn<float>("sharedEnergyFraction",
                            value,
                            "Shared energy over the branch's own energy in the denominator detectors",
                            precision_);
    links->addColumn<float>(
        "score", score, "1 - truth purity: the branch energy the reco object fails to cover (lower is better)", precision_);
  }
  event.put(std::move(links), linkName_ + "Table");
}

void TruthBranchAssociationFlatTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("truthBranchTracksterAssociators", "ticlTrackstersCLUE3DHighRecoToTruthFixed"))
      ->setComment("A ticl::TICLAssociationMap<ticl::mapWithSharedEnergyAndScore> of the branch associators");
  desc.add<edm::InputTag>("targets", edm::InputTag("ticlTruthCBTable", "selected"))
      ->setComment(
          "The particle ids of the truth particle table in row order (the 'selected' product of its "
          "TruthParticleFlatTableProducer), so a link index is a row of that table");
  desc.add<std::string>("direction", "recoToTruth")
      ->setComment("recoToTruth: rows are reco objects and links point at truth rows; truthToReco: the reverse");
  desc.add<std::string>("name", "CLUE3DHighToTruthCB");
  desc.add<std::string>("doc", "ticlTrackstersCLUE3DHigh -> calo-boundary truth association");
  desc.add<bool>("skipNonExistingSrc", true)->setComment("Write empty tables when an input is missing");
  desc.add<int>("precision", -1);
  descriptions.add("truthBranchAssociationFlatTable", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TruthParticleFlatTableProducer);
DEFINE_FWK_MODULE(TruthBranchAssociationFlatTableProducer);
