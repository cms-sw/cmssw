// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#include "PhysicsTools/TruthInfo/interface/Branch.h"

#include <algorithm>
#include <cstring>
#include <queue>
#include <utility>

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

namespace {

  bool isNeutrino(int32_t pdgId) {
    const int32_t id = std::abs(pdgId);
    return id == 12 || id == 14 || id == 16;
  }

  // Ordinary hadron whose quark content includes `flavor` (5 = b, 4 = c).
  bool hadronHasQuark(int32_t pdgId, int32_t flavor) {
    const int32_t id = std::abs(pdgId);
    if (id < 100 || id >= 1000000000)
      return false;
    const int32_t nq1 = (id / 1000) % 10;
    const int32_t nq2 = (id / 100) % 10;
    const int32_t nq3 = (id / 10) % 10;
    return nq1 == flavor || nq2 == flavor || nq3 == flavor;
  }

  // Mirror of TruthGraphProducer::packEventId, which memcpys the EncodedEventId
  // bytes into the low word of a uint64_t. Decode into a trivial uint32_t and
  // rebuild through the public ctor (EncodedEventId is a uint32_t wrapper).
  EncodedEventId decodeEventId(uint64_t packedEventId) {
    uint32_t raw = 0;
    std::memcpy(&raw, &packedEventId, sizeof(raw));
    return EncodedEventId(raw);
  }

}  // namespace

namespace truth {

  Branch::Branch(Graph const* graph, uint32_t rootId, ClosureSpec spec)
      : graph_(graph), roots_{rootId}, spec_(std::move(spec)) {}

  Branch::Branch(Graph const* graph, std::vector<uint32_t> rootIds, ClosureSpec spec)
      : graph_(graph), roots_(std::move(rootIds)), spec_(std::move(spec)) {}

  Particle Branch::root() const { return valid() ? graph_->particle(roots_.front()) : Particle{}; }

  std::vector<Particle> Branch::roots() const {
    std::vector<Particle> out;
    if (!valid())
      return out;
    out.reserve(roots_.size());
    for (uint32_t id : roots_)
      out.push_back(graph_->particle(id));
    return out;
  }

  std::vector<uint32_t> Branch::traverse() const {
    if (!valid())
      return {};

    const uint32_t n = graph_->nParticles();
    std::vector<uint8_t> visited(n, 0);
    std::queue<std::pair<uint32_t, uint32_t>> queue;  // (particleId, depth)
    std::vector<uint32_t> order;

    for (const uint32_t root : roots_) {
      if (root < n && !visited[root]) {
        visited[root] = 1;
        queue.emplace(root, 0);
      }
    }

    while (!queue.empty()) {
      const auto [id, depth] = queue.front();
      queue.pop();
      order.push_back(id);

      bool expand = true;
      switch (spec_.kind) {
        case ClosureKind::DepthN:
          expand = depth < spec_.maxDepth;
          break;
        case ClosureKind::UntilPdgId:
          // Stop at (but include) a particle whose id is in the stop list,
          // unless it is itself a root.
          expand = depth == 0 ||
                   std::find(spec_.stopPdgIds.begin(), spec_.stopPdgIds.end(), graph_->particles()[id].pdgId) ==
                       spec_.stopPdgIds.end();
          break;
        case ClosureKind::Predicate:
          expand = depth == 0 || !(spec_.stopAt && spec_.stopAt(graph_->particle(id)));
          break;
        case ClosureKind::Subtree:
        case ClosureKind::StableLeaves:
          expand = true;
          break;
      }

      if (!expand)
        continue;

      for (const uint32_t vertexId : graph_->decayVertices(id)) {
        if (vertexId >= graph_->nVertices())
          continue;
        for (const uint32_t childId : graph_->outgoingParticles(vertexId)) {
          if (childId < n && !visited[childId]) {
            visited[childId] = 1;
            queue.emplace(childId, depth + 1);
          }
        }
      }
    }

    // For StableLeaves keep only roots and final-state (childless) particles.
    if (spec_.kind == ClosureKind::StableLeaves) {
      const auto isRoot = [this](uint32_t id) { return std::find(roots_.begin(), roots_.end(), id) != roots_.end(); };
      std::erase_if(order, [&](uint32_t id) { return !isRoot(id) && !graph_->particle(id).isLeaf(); });
    }

    std::sort(order.begin(), order.end());
    order.erase(std::unique(order.begin(), order.end()), order.end());
    return order;
  }

  std::vector<uint32_t> Branch::memberIds() const { return traverse(); }

  std::vector<Particle> Branch::members() const {
    std::vector<Particle> out;
    for (uint32_t id : traverse())
      out.push_back(graph_->particle(id));
    return out;
  }

  std::vector<Particle> Branch::stableLeaves() const {
    std::vector<Particle> out;
    if (!valid())
      return out;
    for (uint32_t id : traverse()) {
      auto p = graph_->particle(id);
      if (p.isLeaf())
        out.push_back(p);
    }
    return out;
  }

  math::XYZTLorentzVectorD Branch::p4() const {
    math::XYZTLorentzVectorD sum;
    for (auto const& leaf : stableLeaves())
      sum += leaf.momentum();
    return sum;
  }

  math::XYZTLorentzVectorD Branch::visibleP4() const {
    math::XYZTLorentzVectorD sum;
    for (auto const& leaf : stableLeaves()) {
      if (!isNeutrino(leaf.pdgId()))
        sum += leaf.momentum();
    }
    return sum;
  }

  double Branch::invisibleEnergy() const { return p4().energy() - visibleP4().energy(); }

  int32_t Branch::rootPdgId() const { return valid() ? graph_->particles()[roots_.front()].pdgId : 0; }

  std::optional<Particle> Branch::originWithPdgId(int32_t pdgId) const {
    if (!valid())
      return std::nullopt;
    if (rootPdgId() == pdgId)
      return root();
    return root().firstAncestorWithPdgId(pdgId);
  }

  bool Branch::hasHeavyFlavor(int32_t quarkFlavor) const {
    for (uint32_t id : traverse()) {
      if (hadronHasQuark(graph_->particles()[id].pdgId, quarkFlavor))
        return true;
    }
    return false;
  }

  int32_t Branch::genEvent() const { return valid() ? graph_->particles()[roots_.front()].genEvent : -1; }

  int Branch::bunchCrossing() const {
    return valid() ? decodeEventId(graph_->particles()[roots_.front()].eventId).bunchCrossing() : 0;
  }

  int Branch::event() const { return valid() ? decodeEventId(graph_->particles()[roots_.front()].eventId).event() : 0; }

  std::optional<Particle> Branch::commonAncestor(Branch const& other) const {
    if (!valid() || !other.valid() || graph_ != other.graph_)
      return std::nullopt;
    std::vector<Particle> seeds = roots();
    for (auto const& r : other.roots())
      seeds.push_back(r);
    return graph_->lowestCommonAncestor(seeds);
  }

  Branch Branch::merged(Branch const& other) const {
    if (!valid())
      return other;
    if (!other.valid() || graph_ != other.graph_)
      return *this;
    std::vector<uint32_t> ids = roots_;
    ids.insert(ids.end(), other.roots_.begin(), other.roots_.end());
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return Branch(graph_, std::move(ids), spec_);
  }

}  // namespace truth
