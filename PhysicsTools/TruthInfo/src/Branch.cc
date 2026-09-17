// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "PhysicsTools/TruthInfo/interface/Branch.h"

#include <algorithm>
#include <cstring>
#include <queue>
#include <utility>

#include "FWCore/Utilities/interface/Exception.h"
#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

namespace {

  using truth::isInvisible;

  using truth::hadronHasQuark;

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
      : graph_(graph), roots_{rootId}, spec_(std::move(spec)) {
    validate();
  }

  Branch::Branch(Graph const* graph, std::vector<uint32_t> rootIds, ClosureSpec spec)
      : graph_(graph), roots_(std::move(rootIds)), spec_(std::move(spec)) {
    validate();
  }

  Branch::Branch(Particle const* particle, ClosureSpec spec) : spec_(std::move(spec)) {
    if (particle == nullptr || !particle->valid()) {
      throw cms::Exception("TruthGraphBranch") << "Cannot initialize Branch: particle is invalid.";
    }
    graph_ = particle->graph();
    roots_.push_back(particle->id());
    validate();
  }

  void Branch::validate() {
    if (graph_ == nullptr) {
      throw cms::Exception("TruthGraphBranch") << "Cannot initialize Branch: graph is a nullptr.";
    } else if (roots_.empty()) {
      throw cms::Exception("TruthGraphBranch") << "Cannot initialize Branch: roots list is empty.";
    } else if (std::any_of(roots_.begin(), roots_.end(), [this](uint32_t id) { return id >= graph_->nParticles(); })) {
      throw cms::Exception("TruthGraphBranch") << "Cannot initialize Branch: root particle id does not exist in graph.";
    }
  }

  Particle Branch::root() const { return graph_->particle(roots_.front()); }

  std::vector<Particle> Branch::roots() const {
    std::vector<Particle> out;
    out.reserve(roots_.size());
    for (uint32_t id : roots_)
      out.push_back(graph_->particle(id));
    return out;
  }

  std::vector<uint32_t> Branch::traverse(std::vector<uint32_t>* stopIds) const {
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

      bool stop = false;  // the closure condition fired on this particle
      switch (spec_.kind) {
        case ClosureKind::DepthN:
          stop = depth >= spec_.maxDepth;
          break;
        case ClosureKind::UntilPdgId:
          // Stop at (but include) a particle whose id is in the stop list,
          // unless it is itself a root.
          stop =
              depth > 0 && std::find(spec_.stopPdgIds.begin(), spec_.stopPdgIds.end(), graph_->particles()[id].pdgId) !=
                               spec_.stopPdgIds.end();
          break;
        case ClosureKind::UntilLevels:
          // Stop at (but include) a particle that is at any of the selected truth levels
          stop = (graph_->particles()[id].levelFlags & spec_.levelFlags) != 0;
          break;
        case ClosureKind::Predicate:
          // Stop when predicate condition is satitified (note: includes roots)
          stop = spec_.stopAt && spec_.stopAt(graph_->particle(id));
          break;
        case ClosureKind::Subtree:
        case ClosureKind::StableLeaves:
          stop = graph_->particle(id).isLeaf();  // no decayVertices/children
          break;
      }

      // Stop this chain if the closure condition was met
      if (stop) {
        if (stopIds != nullptr)
          stopIds->push_back(id);
        continue;
      }

      // Add children to queue
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

    // Sort so ids in stopIds and order are ascending
    if (stopIds != nullptr) {
      std::sort(stopIds->begin(), stopIds->end());
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

  std::vector<Particle> Branch::closureLeaves() const {
    std::vector<Particle> out;
    std::vector<uint32_t> stopIds;
    static_cast<void>(traverse(&stopIds));
    out.reserve(stopIds.size());
    for (uint32_t id : stopIds)
      out.push_back(graph_->particle(id));
    return out;
  }

  std::vector<Particle> Branch::stableLeaves() const {
    std::vector<Particle> out;
    for (uint32_t id : traverse()) {
      auto p = graph_->particle(id);
      if (p.isLeaf())
        out.push_back(p);
    }
    return out;
  }

  std::vector<uint32_t> Branch::leaves() const {
    std::vector<uint32_t> ids = traverse();
    if (ids.empty())
      return ids;
    // The members no other member covers, so each particle of the branch is counted
    // once and none of its own ancestors is counted with it. On a Subtree branch these
    // are the final-state leaves. On a TRUNCATED branch they are the particles the
    // closure stopped at, which is the whole point: an UntilPdgId({111}) branch stops at
    // the pi0, whose photons are not members, so the pi0 itself carries the momentum.
    dropCoveredMembers(*graph_, ids, /*keepDeepest=*/true);
    return ids;
  }

  math::XYZTLorentzVectorD Branch::p4() const {
    math::XYZTLorentzVectorD sum;
    for (uint32_t id : leaves())
      sum += graph_->particles()[id].momentum;
    return sum;
  }

  math::XYZTLorentzVectorD Branch::visibleP4() const {
    math::XYZTLorentzVectorD sum;
    for (uint32_t id : leaves()) {
      auto const& particle = graph_->particles()[id];
      if (!isInvisible(particle.pdgId))
        sum += particle.momentum;
    }
    return sum;
  }

  double Branch::invisibleEnergy() const { return p4().energy() - visibleP4().energy(); }

  int32_t Branch::rootPdgId() const { return graph_->particles()[roots_.front()].pdgId; }

  std::optional<Particle> Branch::originWithPdgId(int32_t pdgId) const {
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

  int32_t Branch::genEvent() const { return graph_->particles()[roots_.front()].genEvent; }

  int Branch::bunchCrossing() const {
    return decodeEventId(graph_->particles()[roots_.front()].eventId).bunchCrossing();
  }

  int Branch::event() const { return decodeEventId(graph_->particles()[roots_.front()].eventId).event(); }

  std::optional<Particle> Branch::commonAncestor(Branch const& other) const {
    if (graph_ != other.graph_)
      return std::nullopt;
    std::vector<Particle> seeds = roots();
    for (auto const& r : other.roots())
      seeds.push_back(r);
    return graph_->lowestCommonAncestor(seeds);
  }

  Branch Branch::merged(Branch const& other) const {
    if (graph_ != other.graph_)
      return *this;
    std::vector<uint32_t> ids = roots_;
    ids.insert(ids.end(), other.roots_.begin(), other.roots_.end());
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return Branch(graph_, std::move(ids), spec_);
  }

}  // namespace truth
