// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#include "PhysicsTools/TruthInfo/interface/Graph.h"

#include <algorithm>
#include <cstddef>
#include <queue>
#include <unordered_map>

namespace {
  bool checkCSR(std::vector<uint32_t> const& offsets, std::vector<uint32_t> const& edges, std::size_t nObjects) {
    if (offsets.size() != nObjects + 1)
      return false;
    if (!offsets.empty() && offsets.front() != 0)
      return false;
    if (!offsets.empty() && offsets.back() != edges.size())
      return false;

    for (std::size_t i = 1; i < offsets.size(); ++i) {
      if (offsets[i] < offsets[i - 1])
        return false;
    }
    return true;
  }

  bool checkTargets(std::vector<uint32_t> const& edges, uint32_t limit) {
    for (auto v : edges) {
      if (v >= limit)
        return false;
    }
    return true;
  }
}  // namespace

const truth::ParticleData& truth::Particle::data() const { return graph_->particles.at(id_); }

bool truth::Particle::hasGen() const { return data().hasGen(); }

bool truth::Particle::hasSim() const { return data().hasSim(); }

int32_t truth::Particle::pdgId() const { return data().pdgId; }

int16_t truth::Particle::status() const { return data().status; }

uint64_t truth::Particle::eventId() const { return data().eventId; }

int32_t truth::Particle::genEvent() const { return data().genEvent; }

const math::XYZTLorentzVectorD& truth::Particle::momentum() const { return data().momentum; }

std::span<const truth::Checkpoint> truth::Particle::checkpoints() const {
  return std::span<const truth::Checkpoint>(data().checkpoints.data(), data().checkpoints.size());
}

bool truth::Particle::hasCheckpoints() const { return !data().checkpoints.empty(); }

std::optional<truth::Checkpoint> truth::Particle::checkpoint(uint32_t checkpointId) const {
  for (auto const& cp : data().checkpoints) {
    if (cp.checkpointId == checkpointId)
      return cp;
  }
  return std::nullopt;
}

uint16_t truth::Particle::statusFlags() const { return data().statusFlags; }

bool truth::Particle::isRoot() const { return valid() && graph_->productionVertices(id_).empty(); }

bool truth::Particle::isLeaf() const { return valid() && graph_->decayVertices(id_).empty(); }

std::vector<truth::Vertex> truth::Particle::productionVertices() const {
  return valid() ? graph_->productionVerticesOf(id_) : std::vector<truth::Vertex>{};
}

std::vector<truth::Vertex> truth::Particle::decayVertices() const {
  return valid() ? graph_->decayVerticesOf(id_) : std::vector<truth::Vertex>{};
}

std::vector<truth::Particle> truth::Particle::parents() const {
  return valid() ? graph_->parentsOf(id_) : std::vector<truth::Particle>{};
}

std::vector<truth::Particle> truth::Particle::children() const {
  return valid() ? graph_->childrenOf(id_) : std::vector<truth::Particle>{};
}

std::vector<truth::Particle> truth::Particle::ancestors() const {
  return valid() ? graph_->ancestorsOf(id_) : std::vector<truth::Particle>{};
}

std::vector<truth::Particle> truth::Particle::descendants() const {
  return valid() ? graph_->descendantsOf(id_) : std::vector<truth::Particle>{};
}

bool truth::Particle::hasAncestorPdgId(int pdgId) const { return firstAncestorWithPdgId(pdgId).has_value(); }

std::optional<truth::Particle> truth::Particle::firstAncestorWithPdgId(int pdgId) const {
  return valid() ? graph_->firstAncestorWithPdgIdOf(id_, pdgId) : std::nullopt;
}

std::optional<truth::Particle> truth::Particle::firstCommonAncestor(Particle other) const {
  if (!valid() || !other.valid() || graph_ != other.graph_)
    return std::nullopt;
  return graph_->firstCommonAncestorOf(id_, other.id_);
}

const truth::VertexData& truth::Vertex::data() const { return graph_->vertices.at(id_); }

bool truth::Vertex::hasGen() const { return data().hasGen(); }

bool truth::Vertex::hasSim() const { return data().hasSim(); }

uint64_t truth::Vertex::eventId() const { return data().eventId; }

int32_t truth::Vertex::genEvent() const { return data().genEvent; }

const math::XYZTLorentzVectorD& truth::Vertex::position() const { return data().position; }

bool truth::Vertex::isSource() const { return valid() && graph_->incomingParticles(id_).empty(); }

bool truth::Vertex::isSink() const { return valid() && graph_->outgoingParticles(id_).empty(); }

std::vector<truth::Particle> truth::Vertex::incomingParticles() const {
  return valid() ? graph_->incomingParticlesOf(id_) : std::vector<truth::Particle>{};
}

std::vector<truth::Particle> truth::Vertex::outgoingParticles() const {
  return valid() ? graph_->outgoingParticlesOf(id_) : std::vector<truth::Particle>{};
}

truth::Particle truth::Graph::particle(size_type id) const {
  return id < nParticles() ? Particle(this, id) : Particle{};
}

truth::Vertex truth::Graph::vertex(size_type id) const { return id < nVertices() ? Vertex(this, id) : Vertex{}; }

std::vector<truth::Particle> truth::Graph::particleViews() const {
  std::vector<truth::Particle> out;
  out.reserve(nParticles());
  for (size_type i = 0; i < nParticles(); ++i)
    out.emplace_back(this, i);
  return out;
}

std::vector<truth::Vertex> truth::Graph::vertexViews() const {
  std::vector<truth::Vertex> out;
  out.reserve(nVertices());
  for (size_type i = 0; i < nVertices(); ++i)
    out.emplace_back(this, i);
  return out;
}

std::vector<truth::Particle> truth::Graph::roots() const {
  std::vector<truth::Particle> out;
  for (size_type i = 0; i < nParticles(); ++i) {
    if (productionVertices(i).empty())
      out.emplace_back(this, i);
  }
  return out;
}

std::vector<truth::Particle> truth::Graph::leaves() const {
  std::vector<truth::Particle> out;
  for (size_type i = 0; i < nParticles(); ++i) {
    if (decayVertices(i).empty())
      out.emplace_back(this, i);
  }
  return out;
}

std::vector<truth::Vertex> truth::Graph::sourceVertices() const {
  std::vector<truth::Vertex> out;
  for (size_type i = 0; i < nVertices(); ++i) {
    if (incomingParticles(i).empty())
      out.emplace_back(this, i);
  }
  return out;
}

std::vector<truth::Vertex> truth::Graph::sinkVertices() const {
  std::vector<truth::Vertex> out;
  for (size_type i = 0; i < nVertices(); ++i) {
    if (outgoingParticles(i).empty())
      out.emplace_back(this, i);
  }
  return out;
}

std::vector<truth::Vertex> truth::Graph::productionVerticesOf(size_type particleId) const {
  std::vector<truth::Vertex> out;
  for (uint32_t v : productionVertices(particleId))
    out.emplace_back(this, v);
  return out;
}

std::vector<truth::Vertex> truth::Graph::decayVerticesOf(size_type particleId) const {
  std::vector<truth::Vertex> out;
  for (uint32_t v : decayVertices(particleId))
    out.emplace_back(this, v);
  return out;
}

std::vector<truth::Particle> truth::Graph::incomingParticlesOf(size_type vertexId) const {
  std::vector<truth::Particle> out;
  for (uint32_t p : incomingParticles(vertexId))
    out.emplace_back(this, p);
  return out;
}

std::vector<truth::Particle> truth::Graph::outgoingParticlesOf(size_type vertexId) const {
  std::vector<truth::Particle> out;
  for (uint32_t p : outgoingParticles(vertexId))
    out.emplace_back(this, p);
  return out;
}

void truth::Graph::appendParents(size_type particleId, std::vector<uint32_t>& out) const {
  for (uint32_t v : productionVertices(particleId)) {
    for (uint32_t p : incomingParticles(v)) {
      if (p != particleId)
        out.push_back(p);
    }
  }
}

void truth::Graph::appendChildren(size_type particleId, std::vector<uint32_t>& out) const {
  for (uint32_t v : decayVertices(particleId)) {
    for (uint32_t p : outgoingParticles(v)) {
      if (p != particleId)
        out.push_back(p);
    }
  }
}

namespace {
  // Append the unique particles in `ids` (preserving first-occurrence order) to
  // `out` as views. Degree is tiny, so the O(deg^2) scan beats an nParticles array.
  void appendUnique(truth::Graph const* g, std::vector<uint32_t> const& ids, std::vector<truth::Particle>& out) {
    for (uint32_t p : ids) {
      bool dup = false;
      for (auto const& q : out)
        if (q.id() == p) {
          dup = true;
          break;
        }
      if (!dup)
        out.emplace_back(g, p);
    }
  }
}  // namespace

std::vector<truth::Particle> truth::Graph::parentsOf(size_type particleId) const {
  std::vector<truth::Particle> out;
  if (particleId >= nParticles())
    return out;
  std::vector<uint32_t> ids;
  appendParents(particleId, ids);
  appendUnique(this, ids, out);
  return out;
}

std::vector<truth::Particle> truth::Graph::childrenOf(size_type particleId) const {
  std::vector<truth::Particle> out;
  if (particleId >= nParticles())
    return out;
  std::vector<uint32_t> ids;
  appendChildren(particleId, ids);
  appendUnique(this, ids, out);
  return out;
}

std::vector<truth::Particle> truth::Graph::ancestorsOf(size_type particleId) const {
  std::vector<truth::Particle> out;
  if (particleId >= nParticles())
    return out;

  std::vector<int> dist(nParticles(), -1);
  std::queue<uint32_t> q;
  std::vector<uint32_t> buf;  // reused per-node parent buffer (no per-node alloc)

  appendParents(particleId, buf);
  for (uint32_t p : buf) {
    if (dist[p] >= 0)
      continue;
    dist[p] = 1;
    q.push(p);
    out.emplace_back(this, p);
  }

  while (!q.empty()) {
    const uint32_t cur = q.front();
    q.pop();

    buf.clear();
    appendParents(cur, buf);
    for (uint32_t p : buf) {
      if (dist[p] >= 0)
        continue;
      dist[p] = dist[cur] + 1;
      q.push(p);
      out.emplace_back(this, p);
    }
  }

  return out;
}

std::vector<truth::Particle> truth::Graph::descendantsOf(size_type particleId) const {
  std::vector<truth::Particle> out;
  if (particleId >= nParticles())
    return out;

  std::vector<int> dist(nParticles(), -1);
  std::queue<uint32_t> q;
  std::vector<uint32_t> buf;  // reused per-node child buffer (no per-node alloc)

  appendChildren(particleId, buf);
  for (uint32_t p : buf) {
    if (dist[p] >= 0)
      continue;
    dist[p] = 1;
    q.push(p);
    out.emplace_back(this, p);
  }

  while (!q.empty()) {
    const uint32_t cur = q.front();
    q.pop();

    buf.clear();
    appendChildren(cur, buf);
    for (uint32_t p : buf) {
      if (dist[p] >= 0)
        continue;
      dist[p] = dist[cur] + 1;
      q.push(p);
      out.emplace_back(this, p);
    }
  }

  return out;
}

std::optional<truth::Particle> truth::Graph::firstAncestorWithPdgIdOf(size_type particleId, int pdgId) const {
  if (particleId >= nParticles())
    return std::nullopt;

  std::vector<uint8_t> seen(nParticles(), 0);
  std::queue<uint32_t> q;
  std::vector<uint32_t> buf;  // reused per-node parent buffer (no per-node alloc)

  appendParents(particleId, buf);
  for (uint32_t p : buf) {
    if (seen[p])
      continue;
    seen[p] = 1;
    q.push(p);
  }

  while (!q.empty()) {
    const uint32_t cur = q.front();
    q.pop();

    if (particles[cur].pdgId == pdgId)
      return particle(cur);

    buf.clear();
    appendParents(cur, buf);
    for (uint32_t p : buf) {
      if (seen[p])
        continue;
      seen[p] = 1;
      q.push(p);
    }
  }

  return std::nullopt;
}

std::optional<truth::Particle> truth::Graph::firstCommonAncestorOf(size_type a, size_type b) const {
  if (a >= nParticles() || b >= nParticles())
    return std::nullopt;

  // The two-input lowest common ancestor: same objective (min summed distance,
  // then min worst-case distance, then lowest id).
  return lowestCommonAncestor({particle(a), particle(b)});
}

std::optional<truth::Particle> truth::Graph::lowestCommonAncestor(std::vector<Particle> const& parts) const {
  std::vector<uint32_t> ids;
  ids.reserve(parts.size());
  for (auto const& p : parts) {
    if (p.valid() && p.id() < nParticles())
      ids.push_back(p.id());
  }

  if (ids.empty())
    return std::nullopt;
  if (ids.size() == 1)
    return particle(ids.front());

  // Per-ancestor accumulators keyed only by the ancestors actually visited, so
  // the cost is O(sum of input ancestries) instead of O(inputs x nParticles),
  // with no dense per-input distance matrix and no full-graph scan:
  //   reach  = how many inputs reach the ancestor (common <=> reach == #inputs)
  //   total  = summed upward distance over the inputs
  //   worst  = worst-case (max) upward distance over the inputs
  std::unordered_map<uint32_t, uint32_t> reach;
  std::unordered_map<uint32_t, int> total;
  std::unordered_map<uint32_t, int> worst;

  std::unordered_map<uint32_t, int> dist;  // reused per-input BFS distances (also the visited set)
  std::queue<uint32_t> q;
  std::vector<uint32_t> buf;

  for (const uint32_t start : ids) {
    dist.clear();
    dist.emplace(start, 0);
    q.push(start);

    while (!q.empty()) {
      const uint32_t cur = q.front();
      q.pop();
      const int dcur = dist[cur];

      buf.clear();
      appendParents(cur, buf);
      for (uint32_t p : buf) {
        if (dist.emplace(p, dcur + 1).second)
          q.push(p);
      }
    }

    for (auto const& [node, d] : dist) {
      ++reach[node];
      total[node] += d;
      auto it = worst.find(node);
      if (it == worst.end())
        worst.emplace(node, d);
      else
        it->second = std::max(it->second, d);
    }
  }

  // Among ancestors reached by ALL inputs, pick the closest (min total distance,
  // then min worst-case distance, then lowest id for determinism).
  const uint32_t needed = static_cast<uint32_t>(ids.size());
  int bestId = -1;
  int bestTotal = -1;
  int bestMax = -1;

  for (auto const& [node, r] : reach) {
    if (r != needed)
      continue;

    const int t = total[node];
    const int m = worst[node];
    if (bestId < 0 || t < bestTotal || (t == bestTotal && m < bestMax) ||
        (t == bestTotal && m == bestMax && static_cast<int>(node) < bestId)) {
      bestId = static_cast<int>(node);
      bestTotal = t;
      bestMax = m;
    }
  }

  if (bestId < 0)
    return std::nullopt;

  return particle(static_cast<uint32_t>(bestId));
}

bool truth::Graph::isConsistent() const {
  const bool p2dv = checkCSR(particleToDecayVertexOffsets, particleToDecayVertices, particles.size()) &&
                    checkTargets(particleToDecayVertices, nVertices());

  const bool p2pv = checkCSR(particleToProductionVertexOffsets, particleToProductionVertices, particles.size()) &&
                    checkTargets(particleToProductionVertices, nVertices());

  const bool v2op = checkCSR(vertexToOutgoingParticleOffsets, vertexToOutgoingParticles, vertices.size()) &&
                    checkTargets(vertexToOutgoingParticles, nParticles());

  const bool v2ip = checkCSR(vertexToIncomingParticleOffsets, vertexToIncomingParticles, vertices.size()) &&
                    checkTargets(vertexToIncomingParticles, nParticles());

  return p2dv && p2pv && v2op && v2ip;
}
