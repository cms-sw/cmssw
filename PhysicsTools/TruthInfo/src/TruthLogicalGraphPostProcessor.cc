// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#include "PhysicsTools/TruthInfo/interface/TruthLogicalGraphPostProcessor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace {

  struct DSU {
    std::vector<int> parent;
    std::vector<int> rank;

    explicit DSU(int n) : parent(n), rank(n, 0) {
      for (int i = 0; i < n; ++i)
        parent[i] = i;
    }

    int find(int x) {
      while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    }

    void unite(int a, int b) {
      a = find(a);
      b = find(b);

      if (a == b)
        return;

      if (rank[a] < rank[b])
        std::swap(a, b);

      parent[b] = a;

      if (rank[a] == rank[b])
        ++rank[a];
    }
  };

  bool containsPdgId(std::vector<int32_t> const& pdgIds, int32_t pdgId) {
    return std::find(pdgIds.begin(), pdgIds.end(), pdgId) != pdgIds.end();
  }

  bool containsParticleId(std::vector<uint32_t> const& particleIds, uint32_t particleId) {
    return std::find(particleIds.begin(), particleIds.end(), particleId) != particleIds.end();
  }

  // True if pdgId is an ordinary hadron whose quark content includes the given
  // flavor (5 = b, 4 = c, ...), using the PDG hadron-numbering digits.
  bool hadronHasQuark(int32_t pdgId, int32_t flavor) {
    const int32_t id = std::abs(pdgId);
    if (id < 100 || id >= 1000000000)  // leptons/bosons/diquark-free codes and nuclei are not hadrons here
      return false;
    const int32_t nq1 = (id / 1000) % 10;
    const int32_t nq2 = (id / 100) % 10;
    const int32_t nq3 = (id / 10) % 10;
    return nq1 == flavor || nq2 == flavor || nq3 == flavor;
  }

  bool matchesSeed(truth::Graph const& graph,
                   uint32_t particleId,
                   truth::LogicalGraphPostProcessingConfig const& config) {
    const int32_t pdgId = graph.particles[particleId].pdgId;

    if (containsPdgId(config.seedPdgIds, pdgId))
      return true;

    for (const int32_t flavor : config.seedHadronFlavors) {
      if (hadronHasQuark(pdgId, flavor))
        return true;
    }

    return false;
  }

  bool isIgnoredParticle(truth::Graph const& graph,
                         uint32_t particleId,
                         std::vector<int32_t> const& ignoredPdgIds,
                         std::vector<uint32_t> const& ignoredParticleIds) {
    if (particleId >= graph.nParticles())
      return false;

    if (containsParticleId(ignoredParticleIds, particleId))
      return true;

    if (containsPdgId(ignoredPdgIds, graph.particles[particleId].pdgId))
      return true;

    return false;
  }

  bool isStableGenParticle(truth::Graph const& graph, uint32_t particleId) {
    if (particleId >= graph.nParticles())
      return false;

    auto const& particle = graph.particles[particleId];

    return particle.hasGen() && particle.status == 1;
  }

  bool isGenOnlyVertex(truth::VertexData const& vertex) { return vertex.hasGen() && !vertex.hasSim(); }

  bool isSimOnlyVertex(truth::VertexData const& vertex) { return !vertex.hasGen() && vertex.hasSim(); }

  bool compatibleVertexPositions(truth::VertexData const& a, truth::VertexData const& b, double tolerance) {
    return std::abs(a.position.x() - b.position.x()) <= tolerance &&
           std::abs(a.position.y() - b.position.y()) <= tolerance &&
           std::abs(a.position.z() - b.position.z()) <= tolerance &&
           std::abs(a.position.t() - b.position.t()) <= tolerance;
  }

  bool canMergeGenSimVerticesByPosition(truth::VertexData const& a, truth::VertexData const& b, double tolerance) {
    const bool genSimPair = (isGenOnlyVertex(a) && isSimOnlyVertex(b)) || (isSimOnlyVertex(a) && isGenOnlyVertex(b));

    if (!genSimPair)
      return false;
    const bool areCompatible = compatibleVertexPositions(a, b, tolerance);
    LogDebug("TruthLogicalGraphPostProcessor")
        << "GEN/SIM vertex pair (" << a.position.x() << ", " << a.position.y() << ", " << a.position.z() << ", "
        << a.position.t() << ") vs (" << b.position.x() << ", " << b.position.y() << ", " << b.position.z() << ", "
        << b.position.t() << "): " << (areCompatible ? "merged" : "not merged");
    return areCompatible;
  }

  void buildCSR(uint32_t nSources,
                std::vector<std::pair<uint32_t, uint32_t>>& pairs,
                std::vector<uint32_t>& offsets,
                std::vector<uint32_t>& flat) {
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

    pairs.erase(
        std::remove_if(pairs.begin(), pairs.end(), [nSources](auto const& edge) { return edge.first >= nSources; }),
        pairs.end());

    offsets.assign(nSources + 1, 0);

    for (auto const& edge : pairs) {
      ++offsets[edge.first + 1];
    }

    for (uint32_t i = 1; i <= nSources; ++i) {
      offsets[i] += offsets[i - 1];
    }

    flat.assign(pairs.size(), 0);

    auto cursor = offsets;
    for (auto const& edge : pairs) {
      flat[cursor[edge.first]++] = edge.second;
    }
  }

  // Rebuild the four CSR adjacency arrays of `output` from the edges of
  // `input`, remapping particle and vertex ids and dropping edges with an
  // unmapped endpoint. `extraProductionEdges` are additional
  // (newVertex, newParticle) production-side edges, e.g. those attaching
  // particles to the artificial source vertex. buildCSR sorts and deduplicates,
  // so the collection order here does not affect the result.
  void rebuildAdjacency(truth::Graph const& input,
                        std::vector<int32_t> const& oldParticleToNew,
                        std::vector<int32_t> const& oldVertexToNew,
                        std::vector<std::pair<uint32_t, uint32_t>> const& extraProductionEdges,
                        truth::Graph& output) {
    const uint32_t nParticles = input.nParticles();

    std::vector<std::pair<uint32_t, uint32_t>> particleToDecayVertexPairs;
    std::vector<std::pair<uint32_t, uint32_t>> particleToProductionVertexPairs;
    std::vector<std::pair<uint32_t, uint32_t>> vertexToOutgoingParticlePairs;
    std::vector<std::pair<uint32_t, uint32_t>> vertexToIncomingParticlePairs;

    for (uint32_t oldVertex = 0; oldVertex < input.nVertices(); ++oldVertex) {
      const int32_t newVertex = oldVertexToNew[oldVertex];
      if (newVertex < 0)
        continue;

      for (const uint32_t oldParticle : input.incomingParticles(oldVertex)) {
        if (oldParticle >= nParticles)
          continue;

        const int32_t newParticle = oldParticleToNew[oldParticle];
        if (newParticle < 0)
          continue;

        particleToDecayVertexPairs.emplace_back(static_cast<uint32_t>(newParticle), static_cast<uint32_t>(newVertex));
        vertexToIncomingParticlePairs.emplace_back(static_cast<uint32_t>(newVertex),
                                                   static_cast<uint32_t>(newParticle));
      }

      for (const uint32_t oldParticle : input.outgoingParticles(oldVertex)) {
        if (oldParticle >= nParticles)
          continue;

        const int32_t newParticle = oldParticleToNew[oldParticle];
        if (newParticle < 0)
          continue;

        vertexToOutgoingParticlePairs.emplace_back(static_cast<uint32_t>(newVertex),
                                                   static_cast<uint32_t>(newParticle));
        particleToProductionVertexPairs.emplace_back(static_cast<uint32_t>(newParticle),
                                                     static_cast<uint32_t>(newVertex));
      }
    }

    for (auto const& [newVertex, newParticle] : extraProductionEdges) {
      vertexToOutgoingParticlePairs.emplace_back(newVertex, newParticle);
      particleToProductionVertexPairs.emplace_back(newParticle, newVertex);
    }

    buildCSR(output.nParticles(),
             particleToDecayVertexPairs,
             output.particleToDecayVertexOffsets,
             output.particleToDecayVertices);

    buildCSR(output.nParticles(),
             particleToProductionVertexPairs,
             output.particleToProductionVertexOffsets,
             output.particleToProductionVertices);

    buildCSR(output.nVertices(),
             vertexToOutgoingParticlePairs,
             output.vertexToOutgoingParticleOffsets,
             output.vertexToOutgoingParticles);

    buildCSR(output.nVertices(),
             vertexToIncomingParticlePairs,
             output.vertexToIncomingParticleOffsets,
             output.vertexToIncomingParticles);
  }

  bool directCollapsibleGenParticleChain(truth::Graph const& graph,
                                         uint32_t particleId,
                                         uint32_t& childId,
                                         uint32_t& decayVertexId) {
    if (particleId >= graph.nParticles())
      return false;

    auto const& particle = graph.particles[particleId];

    if (!particle.hasGen())
      return false;

    // Never collapse stable final-state GEN particles.
    if (particle.status == 1)
      return false;

    if (particle.pdgId == 0)
      return false;

    const auto decayVertices = graph.decayVertices(particleId);
    if (decayVertices.size() != 1)
      return false;

    const uint32_t vertexId = decayVertices.front();
    if (vertexId >= graph.nVertices())
      return false;

    auto const& vertex = graph.vertices[vertexId];

    if (!vertex.hasGen() || vertex.hasSim())
      return false;

    const auto incoming = graph.incomingParticles(vertexId);
    const auto outgoing = graph.outgoingParticles(vertexId);

    if (incoming.size() != 1 || incoming.front() != particleId)
      return false;

    if (outgoing.size() != 1)
      return false;

    const uint32_t candidateChild = outgoing.front();
    if (candidateChild >= graph.nParticles())
      return false;

    if (candidateChild == particleId)
      return false;

    auto const& child = graph.particles[candidateChild];

    if (!child.hasGen())
      return false;

    if (child.pdgId != particle.pdgId)
      return false;

    childId = candidateChild;
    decayVertexId = vertexId;

    return true;
  }

  truth::Graph collapseIntermediateGenParticleChains(truth::Graph const& input) {
    if (input.empty())
      return input;

    const uint32_t nParticles = input.nParticles();
    const uint32_t nVertices = input.nVertices();

    std::vector<int32_t> directChild(nParticles, -1);
    std::vector<uint8_t> skipVertex(nVertices, 0);

    for (uint32_t particleId = 0; particleId < nParticles; ++particleId) {
      uint32_t childId = 0;
      uint32_t decayVertexId = 0;

      if (!directCollapsibleGenParticleChain(input, particleId, childId, decayVertexId))
        continue;

      directChild[particleId] = static_cast<int32_t>(childId);
      skipVertex[decayVertexId] = 1;
    }

    std::vector<uint32_t> particleRepresentative(nParticles, 0);

    for (uint32_t particleId = 0; particleId < nParticles; ++particleId) {
      uint32_t current = particleId;

      for (uint32_t step = 0; step < nParticles; ++step) {
        const int32_t next = directChild[current];
        if (next < 0)
          break;

        current = static_cast<uint32_t>(next);
      }

      particleRepresentative[particleId] = current;
    }

    truth::Graph output;

    std::unordered_map<uint32_t, uint32_t> representativeToNewParticle;
    representativeToNewParticle.reserve(nParticles);

    std::vector<int32_t> oldParticleToNew(nParticles, -1);

    for (uint32_t oldParticle = 0; oldParticle < nParticles; ++oldParticle) {
      const uint32_t representative = particleRepresentative[oldParticle];

      auto inserted =
          representativeToNewParticle.emplace(representative, static_cast<uint32_t>(output.particles.size()));

      const uint32_t newParticle = inserted.first->second;

      if (inserted.second) {
        output.particles.push_back(input.particles[representative]);
      }

      oldParticleToNew[oldParticle] = static_cast<int32_t>(newParticle);
    }

    std::vector<uint8_t> keepVertex(nVertices, 0);

    for (uint32_t oldVertex = 0; oldVertex < nVertices; ++oldVertex) {
      if (skipVertex[oldVertex])
        continue;

      for (uint32_t oldParticle : input.incomingParticles(oldVertex)) {
        if (oldParticle < oldParticleToNew.size() && oldParticleToNew[oldParticle] >= 0) {
          keepVertex[oldVertex] = 1;
          break;
        }
      }

      if (keepVertex[oldVertex])
        continue;

      for (uint32_t oldParticle : input.outgoingParticles(oldVertex)) {
        if (oldParticle < oldParticleToNew.size() && oldParticleToNew[oldParticle] >= 0) {
          keepVertex[oldVertex] = 1;
          break;
        }
      }
    }

    std::vector<int32_t> oldVertexToNew(nVertices, -1);

    for (uint32_t oldVertex = 0; oldVertex < nVertices; ++oldVertex) {
      if (!keepVertex[oldVertex])
        continue;

      oldVertexToNew[oldVertex] = static_cast<int32_t>(output.vertices.size());
      output.vertices.push_back(input.vertices[oldVertex]);
    }

    rebuildAdjacency(input, oldParticleToNew, oldVertexToNew, {}, output);

    return output;
  }

  void markDownstreamFromParticle(truth::Graph const& graph,
                                  uint32_t particleId,
                                  std::vector<uint8_t>& keepParticle,
                                  std::vector<uint8_t>& keepVertex) {
    if (particleId >= graph.nParticles())
      return;

    std::queue<uint32_t> queue;

    if (!keepParticle[particleId]) {
      keepParticle[particleId] = 1;
      queue.push(particleId);
    }

    while (!queue.empty()) {
      const uint32_t currentParticle = queue.front();
      queue.pop();

      for (uint32_t vertexId : graph.decayVertices(currentParticle)) {
        if (vertexId >= graph.nVertices())
          continue;

        keepVertex[vertexId] = 1;

        for (uint32_t childId : graph.outgoingParticles(vertexId)) {
          if (childId >= graph.nParticles())
            continue;

          if (!keepParticle[childId]) {
            keepParticle[childId] = 1;
            queue.push(childId);
          }
        }
      }
    }
  }

  // Keep up to parentDepth generations of ancestors above each root as context
  // only: the ancestor particles and the connecting vertices are kept, but
  // their other descendants and their own deeper ancestry are not.
  void markAncestorContext(truth::Graph const& graph,
                           std::vector<uint32_t> const& roots,
                           uint32_t parentDepth,
                           std::vector<uint8_t>& keepParticle,
                           std::vector<uint8_t>& keepVertex) {
    if (parentDepth == 0)
      return;

    std::vector<uint8_t> seen(graph.nParticles(), 0);
    std::queue<std::pair<uint32_t, uint32_t>> queue;

    for (const uint32_t root : roots) {
      if (root >= graph.nParticles())
        continue;

      if (!seen[root]) {
        seen[root] = 1;
        queue.emplace(root, 0);
      }
    }

    while (!queue.empty()) {
      const auto [particleId, depth] = queue.front();
      queue.pop();

      if (depth >= parentDepth)
        continue;

      for (const uint32_t vertexId : graph.productionVertices(particleId)) {
        if (vertexId >= graph.nVertices())
          continue;

        keepVertex[vertexId] = 1;

        for (const uint32_t parentId : graph.incomingParticles(vertexId)) {
          if (parentId >= graph.nParticles())
            continue;

          keepParticle[parentId] = 1;

          if (!seen[parentId]) {
            seen[parentId] = 1;
            queue.emplace(parentId, depth + 1);
          }
        }
      }
    }
  }

  // Restrict matches to the most upstream ones: a match that is a strict
  // descendant of another match is covered by that match's subgraph and is not
  // an independent root. Single multi-source downstream BFS, O(V + E).
  std::vector<uint32_t> mostUpstreamOf(truth::Graph const& graph, std::vector<uint32_t> const& matches) {
    const uint32_t nParticles = graph.nParticles();

    std::vector<uint8_t> strictDescendant(nParticles, 0);
    std::vector<uint8_t> visited(nParticles, 0);
    std::queue<uint32_t> queue;

    for (const uint32_t match : matches) {
      if (match < nParticles && !visited[match]) {
        visited[match] = 1;
        queue.push(match);
      }
    }

    while (!queue.empty()) {
      const uint32_t particleId = queue.front();
      queue.pop();

      for (const uint32_t vertexId : graph.decayVertices(particleId)) {
        if (vertexId >= graph.nVertices())
          continue;

        for (const uint32_t childId : graph.outgoingParticles(vertexId)) {
          if (childId >= nParticles)
            continue;

          strictDescendant[childId] = 1;

          if (!visited[childId]) {
            visited[childId] = 1;
            queue.push(childId);
          }
        }
      }
    }

    std::vector<uint32_t> roots;
    roots.reserve(matches.size());

    for (const uint32_t match : matches) {
      if (match < nParticles && !strictDescendant[match])
        roots.push_back(match);
    }

    return roots;
  }

  // Follow the radiating-copy chain of a particle: while the current copy has
  // exactly one decay vertex with exactly one same-PDG daughter, advance to it.
  // Pure 1 -> 1 copy chains are already gone if collapseIntermediateGenParticles
  // ran before; this handles surviving chains like Z -> Z gamma. Any ambiguity
  // (several decay vertices, several same-PDG daughters) stops the walk.
  uint32_t lastCopyOf(truth::Graph const& graph, uint32_t rootId) {
    const int32_t pdgId = graph.particles[rootId].pdgId;
    uint32_t current = rootId;

    for (uint32_t guard = 0; guard < graph.nParticles(); ++guard) {
      if (graph.particles[current].status == 1)
        break;

      const auto decayVertices = graph.decayVertices(current);
      if (decayVertices.size() != 1)
        break;

      uint32_t sameIdChild = 0;
      uint32_t nSameId = 0;

      for (const uint32_t childId : graph.outgoingParticles(decayVertices.front())) {
        if (childId < graph.nParticles() && childId != current && graph.particles[childId].pdgId == pdgId) {
          sameIdChild = childId;
          ++nSameId;
        }
      }

      if (nSameId != 1)
        break;

      current = sameIdChild;
    }

    return current;
  }

  // Sorted PDG ids of the effective decay products of a root: the outgoing
  // particles of the decay vertices of its last radiating copy.
  std::vector<int32_t> effectiveDecayProductPdgIds(truth::Graph const& graph, uint32_t rootId) {
    const uint32_t lastCopy = lastCopyOf(graph, rootId);

    std::vector<int32_t> pdgIds;

    for (const uint32_t vertexId : graph.decayVertices(lastCopy)) {
      if (vertexId >= graph.nVertices())
        continue;

      for (const uint32_t childId : graph.outgoingParticles(vertexId)) {
        if (childId < graph.nParticles())
          pdgIds.push_back(graph.particles[childId].pdgId);
      }
    }

    std::sort(pdgIds.begin(), pdgIds.end());

    return pdgIds;
  }

  // Multiset containment on sorted ranges: extras in `have` are allowed.
  bool multisetContains(std::vector<int32_t> const& sortedHave, std::vector<int32_t> const& sortedNeed) {
    return std::includes(sortedHave.begin(), sortedHave.end(), sortedNeed.begin(), sortedNeed.end());
  }

  // Direct decay-pattern search: a vertex whose outgoing PDG id multiset
  // contains a group selects that vertex (as common production context) and the
  // matched outgoing particles as roots. Matching is local to one vertex, so
  // unrelated particles from different branches can never be combined.
  void findDecayPatternMatches(truth::Graph const& graph,
                               std::vector<std::vector<int32_t>> const& sortedGroups,
                               std::vector<uint32_t>& roots,
                               std::vector<uint32_t>& matchedVertices) {
    std::vector<int32_t> outgoingPdgIds;

    for (uint32_t vertexId = 0; vertexId < graph.nVertices(); ++vertexId) {
      const auto outgoing = graph.outgoingParticles(vertexId);
      if (outgoing.empty())
        continue;

      outgoingPdgIds.clear();

      for (const uint32_t childId : outgoing) {
        if (childId < graph.nParticles())
          outgoingPdgIds.push_back(graph.particles[childId].pdgId);
      }

      std::sort(outgoingPdgIds.begin(), outgoingPdgIds.end());

      bool vertexMatched = false;

      for (auto const& group : sortedGroups) {
        if (!multisetContains(outgoingPdgIds, group))
          continue;

        vertexMatched = true;

        for (const uint32_t childId : outgoing) {
          if (childId < graph.nParticles() && containsPdgId(group, graph.particles[childId].pdgId))
            roots.push_back(childId);
        }
      }

      if (vertexMatched)
        matchedVertices.push_back(vertexId);
    }

    std::sort(roots.begin(), roots.end());
    roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
  }

  void dropVerticesWithoutVisibleParticles(truth::Graph const& graph,
                                           std::vector<uint8_t> const& keepParticle,
                                           std::vector<uint8_t>& keepVertex) {
    for (uint32_t vertexId = 0; vertexId < graph.nVertices(); ++vertexId) {
      if (!keepVertex[vertexId])
        continue;

      bool hasVisibleIncoming = false;
      bool hasVisibleOutgoing = false;

      for (uint32_t particleId : graph.incomingParticles(vertexId)) {
        if (particleId < graph.nParticles() && keepParticle[particleId]) {
          hasVisibleIncoming = true;
          break;
        }
      }

      for (uint32_t particleId : graph.outgoingParticles(vertexId)) {
        if (particleId < graph.nParticles() && keepParticle[particleId]) {
          hasVisibleOutgoing = true;
          break;
        }
      }

      if (!hasVisibleIncoming && !hasVisibleOutgoing)
        keepVertex[vertexId] = 0;
    }
  }

  // attachRole[i] is 0 for particles that are not attached to an artificial
  // source, or the uint8_t value of a non-Normal VertexRole otherwise. One
  // artificial source vertex is created per (role, genEvent) so that overlaid
  // pile-up graphs stay distinguishable; it carries the genEvent/eventId of the
  // activity it summarizes.
  truth::Graph rebuildFilteredGraph(truth::Graph const& input,
                                    std::vector<uint8_t> const& keepParticle,
                                    std::vector<uint8_t> const& keepVertex,
                                    std::vector<uint8_t> const& attachRole) {
    const uint32_t nParticles = input.nParticles();
    const uint32_t nVertices = input.nVertices();

    truth::Graph output;

    std::vector<int32_t> oldParticleToNew(nParticles, -1);
    std::vector<int32_t> oldVertexToNew(nVertices, -1);

    output.particles.reserve(nParticles);
    output.vertices.reserve(nVertices + 2);

    for (uint32_t oldParticle = 0; oldParticle < nParticles; ++oldParticle) {
      if (!keepParticle[oldParticle])
        continue;

      oldParticleToNew[oldParticle] = static_cast<int32_t>(output.particles.size());
      output.particles.push_back(input.particles[oldParticle]);
    }

    for (uint32_t oldVertex = 0; oldVertex < nVertices; ++oldVertex) {
      if (!keepVertex[oldVertex])
        continue;

      oldVertexToNew[oldVertex] = static_cast<int32_t>(output.vertices.size());
      output.vertices.push_back(input.vertices[oldVertex]);
    }

    std::map<std::pair<uint8_t, int32_t>, uint32_t> artificialNode;
    std::vector<std::pair<uint32_t, uint32_t>> artificialEdges;

    for (uint32_t oldParticle = 0; oldParticle < nParticles; ++oldParticle) {
      const uint8_t role = attachRole[oldParticle];
      if (role == 0)
        continue;

      const int32_t newParticle = oldParticleToNew[oldParticle];
      if (newParticle < 0)
        continue;

      const int32_t genEvent = input.particles[oldParticle].genEvent;
      const std::pair<uint8_t, int32_t> key{role, genEvent};

      auto it = artificialNode.find(key);
      if (it == artificialNode.end()) {
        truth::VertexData vertex;
        vertex.genNode = -1;
        vertex.simNode = -1;
        vertex.role = role;
        vertex.genEvent = genEvent;
        vertex.eventId = input.particles[oldParticle].eventId;

        it = artificialNode.emplace(key, static_cast<uint32_t>(output.vertices.size())).first;
        output.vertices.push_back(vertex);
      }

      artificialEdges.emplace_back(it->second, static_cast<uint32_t>(newParticle));
    }

    rebuildAdjacency(input, oldParticleToNew, oldVertexToNew, artificialEdges, output);

    return output;
  }

  truth::Graph filterGraphBySelection(truth::Graph const& input,
                                      truth::LogicalGraphPostProcessingConfig const& config) {
    if (input.empty())
      return input;

    // Skip empty groups: they would match every vertex.
    std::vector<std::vector<int32_t>> sortedGroups;
    sortedGroups.reserve(config.decayPdgIdGroups.size());

    for (auto const& group : config.decayPdgIdGroups) {
      if (!group.empty()) {
        sortedGroups.push_back(group);
        std::sort(sortedGroups.back().begin(), sortedGroups.back().end());
      }
    }

    const bool haveSeeds = !config.seedPdgIds.empty() || !config.seedHadronFlavors.empty();
    const bool haveGroups = !sortedGroups.empty();

    if (!haveSeeds && !haveGroups)
      return input;

    // Debug escape hatch: no real particle has PDG id 0, so seedPdgIds = {0}
    // explicitly requests the full, unfiltered graph.
    if (containsPdgId(config.seedPdgIds, 0))
      return input;

    const uint32_t nParticles = input.nParticles();
    const uint32_t nVertices = input.nVertices();

    std::vector<uint32_t> roots;
    std::vector<uint32_t> patternVertices;

    if (haveSeeds) {
      std::vector<uint32_t> matches;

      for (uint32_t particleId = 0; particleId < nParticles; ++particleId) {
        if (matchesSeed(input, particleId, config))
          matches.push_back(particleId);
      }

      if (!matches.empty()) {
        roots = mostUpstreamOf(input, matches);

        if (haveGroups) {
          // Keep only seed roots whose effective decay matches a group, e.g.
          // Z -> mu+ mu- but not Z -> e+ e-.
          std::erase_if(roots, [&](uint32_t root) {
            const auto products = effectiveDecayProductPdgIds(input, root);
            return std::none_of(sortedGroups.begin(), sortedGroups.end(), [&](auto const& group) {
              return multisetContains(products, group);
            });
          });
        }
      } else if (haveGroups) {
        // The generator did not write the requested resonance explicitly:
        // fall back to the direct decay-pattern search.
        findDecayPatternMatches(input, sortedGroups, roots, patternVertices);
      }
    } else {
      findDecayPatternMatches(input, sortedGroups, roots, patternVertices);
    }

    if (roots.empty()) {
      edm::LogWarning("TruthLogicalGraphPostProcessor")
          << "Configured truth graph selection (seedPdgIds and/or decayPdgIdGroups) matched nothing in this event; "
          << (config.keepStableSpectators ? "keeping only stable GEN particles as underlying-event spectators."
                                          : "the selected graph is empty.");
    }

    std::vector<uint8_t> keepParticle(nParticles, 0);
    std::vector<uint8_t> keepVertex(nVertices, 0);

    for (const uint32_t root : roots) {
      markDownstreamFromParticle(input, root, keepParticle, keepVertex);
    }

    // Pattern-matched vertices are kept as the common production context of
    // their matched outgoing particles.
    for (const uint32_t vertexId : patternVertices) {
      keepVertex[vertexId] = 1;
    }

    markAncestorContext(input, roots, config.seedParentDepth, keepParticle, keepVertex);

    // Optionally keep every stable final-state GEN particle outside the
    // selection; these become the underlying-event spectators. Disabled by
    // keepStableSpectators=false for a focused subgraph.
    std::vector<uint8_t> stableSpectator(nParticles, 0);

    if (config.keepStableSpectators) {
      for (uint32_t particleId = 0; particleId < nParticles; ++particleId) {
        if (!isStableGenParticle(input, particleId))
          continue;

        if (isIgnoredParticle(input, particleId, config.ignoredPdgIds, config.ignoredParticleIds))
          continue;

        if (keepParticle[particleId])
          continue;

        keepParticle[particleId] = 1;
        stableSpectator[particleId] = 1;
      }
    }

    dropVerticesWithoutVisibleParticles(input, keepParticle, keepVertex);

    // Assign an artificial-source role to every kept particle whose real
    // production vertices were all dropped: stable spectators -> UnderlyingEvent,
    // selected roots / truncated ancestors at the upstream boundary -> Upstream
    // (ISR). True sources of the input graph stay sources.
    std::vector<uint8_t> attachRole(nParticles, 0);

    for (uint32_t particleId = 0; particleId < nParticles; ++particleId) {
      if (!keepParticle[particleId])
        continue;

      const auto productionVertices = input.productionVertices(particleId);

      if (productionVertices.empty() && !stableSpectator[particleId])
        continue;

      const bool hasKeptProduction =
          std::any_of(productionVertices.begin(), productionVertices.end(), [&](uint32_t vertexId) {
            return vertexId < nVertices && keepVertex[vertexId];
          });

      if (!hasKeptProduction) {
        attachRole[particleId] = static_cast<uint8_t>(stableSpectator[particleId] ? truth::VertexRole::UnderlyingEvent
                                                                                  : truth::VertexRole::Upstream);
      }
    }

    return rebuildFilteredGraph(input, keepParticle, keepVertex, attachRole);
  }

  void mergeVertexPayload(truth::VertexData& output, truth::VertexData const& input) {
    if (input.hasGen()) {
      output.genNode = input.genNode;
      output.genEvent = input.genEvent;

      // Prefer the GEN position for merged GEN+SIM vertices.
      output.position = input.position;
    }

    if (input.hasSim()) {
      output.simNode = input.simNode;
      output.eventId = input.eventId;

      if (!output.hasGen()) {
        output.position = input.position;
      }
    }
  }

  truth::Graph mergeGenSimVerticesByPosition(truth::Graph const& input, double tolerance) {
    if (input.empty())
      return input;

    const uint32_t nParticles = input.nParticles();
    const uint32_t nVertices = input.nVertices();

    if (nVertices == 0)
      return input;

    DSU vertexDSU(static_cast<int>(nVertices));

    auto tryMergeVertexList = [&](std::vector<truth::Vertex> const& vertices) {
      for (std::size_t i = 0; i < vertices.size(); ++i) {
        const uint32_t first = vertices[i].id();

        if (first >= nVertices)
          continue;

        for (std::size_t j = i + 1; j < vertices.size(); ++j) {
          const uint32_t second = vertices[j].id();

          if (second >= nVertices)
            continue;

          if (canMergeGenSimVerticesByPosition(input.vertices[first], input.vertices[second], tolerance)) {
            vertexDSU.unite(static_cast<int>(first), static_cast<int>(second));
          }
        }
      }
    };

    for (uint32_t particleId = 0; particleId < nParticles; ++particleId) {
      tryMergeVertexList(input.particle(particleId).productionVertices());
      tryMergeVertexList(input.particle(particleId).decayVertices());
    }

    bool changed = false;

    for (uint32_t vertexId = 0; vertexId < nVertices; ++vertexId) {
      if (vertexDSU.find(static_cast<int>(vertexId)) != static_cast<int>(vertexId)) {
        changed = true;
        break;
      }
    }

    if (!changed)
      return input;

    truth::Graph output;

    output.particles = input.particles;

    std::unordered_map<int, uint32_t> vertexRepToNew;
    vertexRepToNew.reserve(nVertices);

    std::vector<int32_t> oldVertexToNew(nVertices, -1);

    for (uint32_t oldVertex = 0; oldVertex < nVertices; ++oldVertex) {
      const int rep = vertexDSU.find(static_cast<int>(oldVertex));

      auto inserted = vertexRepToNew.emplace(rep, static_cast<uint32_t>(output.vertices.size()));
      const uint32_t newVertex = inserted.first->second;

      if (inserted.second) {
        output.vertices.emplace_back();
      }

      oldVertexToNew[oldVertex] = static_cast<int32_t>(newVertex);
      mergeVertexPayload(output.vertices[newVertex], input.vertices[oldVertex]);
    }

    // Particles are untouched by vertex merging: identity map.
    std::vector<int32_t> oldParticleToNew(nParticles);
    std::iota(oldParticleToNew.begin(), oldParticleToNew.end(), 0);

    rebuildAdjacency(input, oldParticleToNew, oldVertexToNew, {}, output);

    return output;
  }

  truth::Graph collapseIgnoredParticles(truth::Graph const& input,
                                        std::vector<int32_t> const& ignoredPdgIds,
                                        std::vector<uint32_t> const& ignoredParticleIds) {
    if (input.empty())
      return input;

    if (ignoredPdgIds.empty() && ignoredParticleIds.empty())
      return input;

    const uint32_t nParticles = input.nParticles();
    const uint32_t nVertices = input.nVertices();

    std::vector<uint8_t> removeParticle(nParticles, 0);

    for (uint32_t particleId = 0; particleId < nParticles; ++particleId) {
      if (isIgnoredParticle(input, particleId, ignoredPdgIds, ignoredParticleIds))
        removeParticle[particleId] = 1;
    }

    if (std::none_of(removeParticle.begin(), removeParticle.end(), [](uint8_t value) { return value != 0; }))
      return input;

    DSU vertexDSU(static_cast<int>(nVertices));

    for (uint32_t particleId = 0; particleId < nParticles; ++particleId) {
      if (!removeParticle[particleId])
        continue;

      std::vector<uint32_t> connectedVertices;

      for (uint32_t vertexId : input.productionVertices(particleId)) {
        if (vertexId < nVertices)
          connectedVertices.push_back(vertexId);
      }

      for (uint32_t vertexId : input.decayVertices(particleId)) {
        if (vertexId < nVertices)
          connectedVertices.push_back(vertexId);
      }

      if (connectedVertices.size() < 2)
        continue;

      const uint32_t first = connectedVertices.front();

      for (std::size_t i = 1; i < connectedVertices.size(); ++i) {
        vertexDSU.unite(static_cast<int>(first), static_cast<int>(connectedVertices[i]));
      }
    }

    truth::Graph output;

    std::vector<int32_t> oldParticleToNew(nParticles, -1);

    for (uint32_t oldParticle = 0; oldParticle < nParticles; ++oldParticle) {
      if (removeParticle[oldParticle])
        continue;

      oldParticleToNew[oldParticle] = static_cast<int32_t>(output.particles.size());
      output.particles.push_back(input.particles[oldParticle]);
    }

    std::unordered_map<int, uint32_t> vertexRepToNew;
    vertexRepToNew.reserve(nVertices);

    std::vector<int32_t> oldVertexToNew(nVertices, -1);

    // Vertices in one DSU group share the id of the first visible member;
    // vertices with no visible particle at all stay unmapped and disappear.
    for (uint32_t oldVertex = 0; oldVertex < nVertices; ++oldVertex) {
      const auto hasVisible = [&](auto const& particles) {
        return std::any_of(particles.begin(), particles.end(), [&](uint32_t oldParticle) {
          return oldParticle < nParticles && oldParticleToNew[oldParticle] >= 0;
        });
      };

      if (!hasVisible(input.incomingParticles(oldVertex)) && !hasVisible(input.outgoingParticles(oldVertex)))
        continue;

      const int rep = vertexDSU.find(static_cast<int>(oldVertex));
      auto inserted = vertexRepToNew.emplace(rep, static_cast<uint32_t>(output.vertices.size()));

      if (inserted.second) {
        output.vertices.push_back(input.vertices[oldVertex]);
      }

      oldVertexToNew[oldVertex] = static_cast<int32_t>(inserted.first->second);
    }

    rebuildAdjacency(input, oldParticleToNew, oldVertexToNew, {}, output);

    return output;
  }

}  // namespace

namespace truth {

  TruthLogicalGraphPostProcessor::TruthLogicalGraphPostProcessor(LogicalGraphPostProcessingConfig config)
      : config_(std::move(config)) {}

  edm::ParameterSetDescription TruthLogicalGraphPostProcessor::psetDescription() {
    edm::ParameterSetDescription desc;

    desc.add<bool>("collapseIntermediateGenParticles", true)
        ->setComment(
            "If true, collapse GEN chains P -> V -> C where P has status != 1, C is the only daughter, "
            "and P and C have the same PDG id. Status-1 GEN particles are never collapsed by this rule.");

    desc.add<std::vector<int32_t>>("seedPdgIds", {})
        ->setComment(
            "If non-empty, particles with these exact PDG ids seed the selection: the most upstream particle of "
            "each matching chain becomes a root and its full downstream subgraph is kept. The special value 0 "
            "disables the selection and keeps the full graph (debugging). Stable GEN particles outside the "
            "selection are kept and attached to one artificial source vertex.");

    desc.add<uint32_t>("seedParentDepth", 0)
        ->setComment(
            "Number of ancestor generations kept above each selected root as context only: the ancestors and "
            "connecting vertices are kept, but not their other descendants. Kept particles whose production "
            "vertices all fall outside the selection are attached to an artificial Upstream (ISR) source vertex.");

    desc.add<std::vector<int32_t>>("seedHadronFlavors", {})
        ->setComment(
            "Seed on hadrons by heavy-flavor content (5 = b, 4 = c): a hadron whose quark content includes any of "
            "these flavors becomes a seed, e.g. {5} selects all B-hadron decay subgraphs. OR-ed with seedPdgIds.");

    desc.add<bool>("keepStableSpectators", true)
        ->setComment(
            "If true, stable final-state GEN particles outside the selected subgraph are kept and attached to an "
            "artificial UnderlyingEvent source vertex (tagged with their genEvent/eventId for pile-up provenance). "
            "If false, they are dropped, giving a focused subgraph with only the selection and its Upstream (ISR) "
            "context. Only meaningful when a selection (seedPdgIds/decayPdgIdGroups) is active.");

    {
      edm::ParameterSetDescription groupDesc;
      groupDesc.add<std::vector<int32_t>>("pdgIds", {})
          ->setComment("Unordered, charge-sensitive multiset of required PDG ids, e.g. (13, -13).");

      desc.addVPSet("decayPdgIdGroups", groupDesc, {})
          ->setComment(
              "Decay patterns of interest; groups are OR-ed. Without seedPdgIds: a vertex whose outgoing PDG ids "
              "contain a group as a sub-multiset is selected and the matched particles plus their downstream "
              "subgraphs are kept. With seedPdgIds: only seed roots whose effective decay products (after following "
              "same-PDG radiating copy chains) contain a group are kept; if the event has no particle with a seed "
              "PDG id at all, the direct vertex search is used as a fallback.");
    }

    desc.add<std::vector<int32_t>>("ignoredPdgIds", {})
        ->setComment(
            "Particles with these exact PDG ids are always removed from the final logical graph. If internal, "
            "their production and decay vertices are merged so the graph remains connected.");

    desc.add<std::vector<uint32_t>>("ignoredParticleIds", {})
        ->setComment(
            "Logical particle ids to remove from the final logical graph. These ids refer to the graph state at "
            "the moment the ignored-particle collapsing step is applied.");

    desc.add<bool>("mergeGenSimVerticesByPosition", true)
        ->setComment(
            "If true, merge GEN-only and SIM-only logical vertices connected to the same particle when their "
            "positions are compatible within genSimVertexPositionTolerance.");

    desc.add<double>("genSimVertexPositionTolerance", 5e-3)
        ->setComment("Absolute tolerance applied independently to x, y, z, and t when merging GEN and SIM vertices.");

    return desc;
  }

  LogicalGraphPostProcessingConfig TruthLogicalGraphPostProcessor::configFromPSet(edm::ParameterSet const& pset) {
    LogicalGraphPostProcessingConfig config;

    config.collapseIntermediateGenParticles = pset.getParameter<bool>("collapseIntermediateGenParticles");
    config.seedPdgIds = pset.getParameter<std::vector<int32_t>>("seedPdgIds");
    config.seedHadronFlavors = pset.getParameter<std::vector<int32_t>>("seedHadronFlavors");
    config.seedParentDepth = pset.getParameter<uint32_t>("seedParentDepth");
    config.keepStableSpectators = pset.getParameter<bool>("keepStableSpectators");

    for (auto const& groupPSet : pset.getParameter<std::vector<edm::ParameterSet>>("decayPdgIdGroups")) {
      config.decayPdgIdGroups.push_back(groupPSet.getParameter<std::vector<int32_t>>("pdgIds"));
    }
    config.ignoredPdgIds = pset.getParameter<std::vector<int32_t>>("ignoredPdgIds");
    config.ignoredParticleIds = pset.getParameter<std::vector<uint32_t>>("ignoredParticleIds");
    config.mergeGenSimVerticesByPosition = pset.getParameter<bool>("mergeGenSimVerticesByPosition");
    config.genSimVertexPositionTolerance = pset.getParameter<double>("genSimVertexPositionTolerance");

    return config;
  }

  Graph TruthLogicalGraphPostProcessor::process(Graph input) const {
    if (config_.collapseIntermediateGenParticles) {
      input = collapseIntermediateGenParticleChains(input);
    }

    input = filterGraphBySelection(input, config_);

    if (config_.mergeGenSimVerticesByPosition) {
      input = mergeGenSimVerticesByPosition(input, config_.genSimVertexPositionTolerance);
    }

    if (!config_.ignoredPdgIds.empty() || !config_.ignoredParticleIds.empty()) {
      input = collapseIgnoredParticles(input, config_.ignoredPdgIds, config_.ignoredParticleIds);
    }

    return input;
  }

}  // namespace truth
