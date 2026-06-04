#include "PhysicsTools/TruthInfo/interface/TruthLogicalGraphPostProcessor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

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
    return std::abs(a.position.px() - b.position.px()) <= tolerance &&
           std::abs(a.position.py() - b.position.py()) <= tolerance &&
           std::abs(a.position.pz() - b.position.pz()) <= tolerance &&
           std::abs(a.position.e() - b.position.e()) <= tolerance;
  }

  bool canMergeGenSimVerticesByPosition(truth::VertexData const& a, truth::VertexData const& b, double tolerance) {
    const bool genSimPair = (isGenOnlyVertex(a) && isSimOnlyVertex(b)) || (isSimOnlyVertex(a) && isGenOnlyVertex(b));

    if (!genSimPair)
      return false;
    bool areCompatible = compatibleVertexPositions(a, b, tolerance);
    if (!areCompatible) {
      std::cout << "vertices not merged: (" << a.position.px() << " , " << a.position.py() << " , " << a.position.pz()
                << ") too far from " << "(" << b.position.px() << " , " << b.position.py() << " , " << b.position.pz()
                << ")" << std::endl;
    } else {
      std::cout << "vertices successfully merged: (" << a.position.px() << " , " << a.position.py() << " , "
                << a.position.pz() << ") too far from " << "(" << b.position.px() << " , " << b.position.py() << " , "
                << b.position.pz() << ")" << std::endl;
    }
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

    std::vector<std::pair<uint32_t, uint32_t>> particleToDecayVertexPairs;
    std::vector<std::pair<uint32_t, uint32_t>> particleToProductionVertexPairs;
    std::vector<std::pair<uint32_t, uint32_t>> vertexToOutgoingParticlePairs;
    std::vector<std::pair<uint32_t, uint32_t>> vertexToIncomingParticlePairs;

    for (uint32_t oldVertex = 0; oldVertex < nVertices; ++oldVertex) {
      const int32_t newVertex = oldVertexToNew[oldVertex];
      if (newVertex < 0)
        continue;

      for (uint32_t oldParticle : input.incomingParticles(oldVertex)) {
        if (oldParticle >= oldParticleToNew.size())
          continue;

        const int32_t newParticle = oldParticleToNew[oldParticle];
        if (newParticle < 0)
          continue;

        particleToDecayVertexPairs.emplace_back(static_cast<uint32_t>(newParticle), static_cast<uint32_t>(newVertex));
        vertexToIncomingParticlePairs.emplace_back(static_cast<uint32_t>(newVertex),
                                                   static_cast<uint32_t>(newParticle));
      }

      for (uint32_t oldParticle : input.outgoingParticles(oldVertex)) {
        if (oldParticle >= oldParticleToNew.size())
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

  void markUpstreamToParticle(truth::Graph const& graph,
                              uint32_t particleId,
                              std::vector<uint8_t>& keepParticle,
                              std::vector<uint8_t>& keepVertex) {
    if (particleId >= graph.nParticles())
      return;

    std::queue<uint32_t> queue;

    if (!keepParticle[particleId]) {
      keepParticle[particleId] = 1;
    }

    queue.push(particleId);

    while (!queue.empty()) {
      const uint32_t currentParticle = queue.front();
      queue.pop();

      for (uint32_t vertexId : graph.productionVertices(currentParticle)) {
        if (vertexId >= graph.nVertices())
          continue;

        keepVertex[vertexId] = 1;

        for (uint32_t parentId : graph.incomingParticles(vertexId)) {
          if (parentId >= graph.nParticles())
            continue;

          if (!keepParticle[parentId]) {
            keepParticle[parentId] = 1;
            queue.push(parentId);
          }
        }
      }
    }
  }

  std::vector<uint32_t> expandSeedsWithParents(truth::Graph const& graph,
                                               std::vector<uint32_t> const& seedParticles,
                                               uint32_t parentDepth) {
    std::vector<uint32_t> startParticles = seedParticles;

    std::vector<uint8_t> seen(graph.nParticles(), 0);
    std::queue<std::pair<uint32_t, uint32_t>> queue;

    for (const uint32_t seed : seedParticles) {
      if (seed >= graph.nParticles())
        continue;

      if (!seen[seed]) {
        seen[seed] = 1;
        queue.emplace(seed, 0);
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

        for (const uint32_t parentId : graph.incomingParticles(vertexId)) {
          if (parentId >= graph.nParticles())
            continue;

          if (!seen[parentId]) {
            seen[parentId] = 1;
            startParticles.push_back(parentId);
            queue.emplace(parentId, depth + 1);
          }
        }
      }
    }

    std::sort(startParticles.begin(), startParticles.end());
    startParticles.erase(std::unique(startParticles.begin(), startParticles.end()), startParticles.end());

    return startParticles;
  }

  void closeKeptVerticesWithAllParents(truth::Graph const& graph,
                                       std::vector<uint8_t>& keepParticle,
                                       std::vector<uint8_t>& keepVertex) {
    bool changed = true;

    while (changed) {
      changed = false;

      for (uint32_t vertexId = 0; vertexId < graph.nVertices(); ++vertexId) {
        if (!keepVertex[vertexId])
          continue;

        for (uint32_t parentId : graph.incomingParticles(vertexId)) {
          if (parentId >= graph.nParticles())
            continue;

          if (keepParticle[parentId])
            continue;

          markUpstreamToParticle(graph, parentId, keepParticle, keepVertex);
          changed = true;
        }
      }
    }
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

  truth::Graph rebuildFilteredGraph(truth::Graph const& input,
                                    std::vector<uint8_t> const& keepParticle,
                                    std::vector<uint8_t> const& keepVertex,
                                    std::vector<uint8_t> const& connectToCollapsedStableVertex) {
    const uint32_t nParticles = input.nParticles();
    const uint32_t nVertices = input.nVertices();

    const bool needCollapsedStableVertex = std::any_of(connectToCollapsedStableVertex.begin(),
                                                       connectToCollapsedStableVertex.end(),
                                                       [](uint8_t value) { return value != 0; });

    truth::Graph output;

    std::vector<int32_t> oldParticleToNew(nParticles, -1);
    std::vector<int32_t> oldVertexToNew(nVertices, -1);

    output.particles.reserve(nParticles);
    output.vertices.reserve(nVertices + (needCollapsedStableVertex ? 1 : 0));

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

    int32_t collapsedStableVertex = -1;

    if (needCollapsedStableVertex) {
      collapsedStableVertex = static_cast<int32_t>(output.vertices.size());

      truth::VertexData vertex;
      vertex.genNode = -1;
      vertex.simNode = -1;
      vertex.eventId = 0;
      vertex.genEvent = -1;

      output.vertices.push_back(vertex);
    }

    std::vector<std::pair<uint32_t, uint32_t>> particleToDecayVertexPairs;
    std::vector<std::pair<uint32_t, uint32_t>> particleToProductionVertexPairs;
    std::vector<std::pair<uint32_t, uint32_t>> vertexToOutgoingParticlePairs;
    std::vector<std::pair<uint32_t, uint32_t>> vertexToIncomingParticlePairs;

    for (uint32_t oldVertex = 0; oldVertex < nVertices; ++oldVertex) {
      const int32_t newVertex = oldVertexToNew[oldVertex];
      if (newVertex < 0)
        continue;

      for (uint32_t oldParticle : input.incomingParticles(oldVertex)) {
        if (oldParticle >= nParticles)
          continue;

        const int32_t newParticle = oldParticleToNew[oldParticle];
        if (newParticle < 0)
          continue;

        particleToDecayVertexPairs.emplace_back(static_cast<uint32_t>(newParticle), static_cast<uint32_t>(newVertex));
        vertexToIncomingParticlePairs.emplace_back(static_cast<uint32_t>(newVertex),
                                                   static_cast<uint32_t>(newParticle));
      }

      for (uint32_t oldParticle : input.outgoingParticles(oldVertex)) {
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

    if (collapsedStableVertex >= 0) {
      const uint32_t newVertex = static_cast<uint32_t>(collapsedStableVertex);

      for (uint32_t oldParticle = 0; oldParticle < nParticles; ++oldParticle) {
        if (!connectToCollapsedStableVertex[oldParticle])
          continue;

        const int32_t newParticle = oldParticleToNew[oldParticle];
        if (newParticle < 0)
          continue;

        vertexToOutgoingParticlePairs.emplace_back(newVertex, static_cast<uint32_t>(newParticle));
        particleToProductionVertexPairs.emplace_back(static_cast<uint32_t>(newParticle), newVertex);
      }
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

    return output;
  }

  truth::Graph filterGraphBySeedPdgIds(truth::Graph const& input,
                                       truth::LogicalGraphPostProcessingConfig const& config) {
    if (input.empty())
      return input;

    if (config.seedPdgIds.empty())
      return input;

    const uint32_t nParticles = input.nParticles();
    const uint32_t nVertices = input.nVertices();

    std::vector<uint32_t> seedParticles;

    for (uint32_t particleId = 0; particleId < nParticles; ++particleId) {
      if (containsPdgId(config.seedPdgIds, input.particles[particleId].pdgId))
        seedParticles.push_back(particleId);
    }

    if (seedParticles.empty())
      return input;

    const auto startParticles = expandSeedsWithParents(input, seedParticles, config.seedParentDepth);

    std::vector<uint8_t> keepParticle(nParticles, 0);
    std::vector<uint8_t> keepVertex(nVertices, 0);

    for (const uint32_t particleId : startParticles) {
      markDownstreamFromParticle(input, particleId, keepParticle, keepVertex);
    }

    // The graph is a DAG, not a tree. If a kept downstream vertex has parents
    // that are not in the selected subgraph, keep those parents and their full
    // upstream history to avoid creating misleading partial vertices.
    closeKeptVerticesWithAllParents(input, keepParticle, keepVertex);

    // Keep every final-state GEN particle unless the user explicitly asked to
    // ignore/collapse it. Stable particles outside the interesting subgraph are
    // attached to one artificial source vertex.
    std::vector<uint8_t> connectToCollapsedStableVertex(nParticles, 0);

    for (uint32_t particleId = 0; particleId < nParticles; ++particleId) {
      if (!isStableGenParticle(input, particleId))
        continue;

      if (isIgnoredParticle(input, particleId, config.ignoredPdgIds, config.ignoredParticleIds))
        continue;

      if (keepParticle[particleId])
        continue;

      keepParticle[particleId] = 1;
      connectToCollapsedStableVertex[particleId] = 1;
    }

    dropVerticesWithoutVisibleParticles(input, keepParticle, keepVertex);

    return rebuildFilteredGraph(input, keepParticle, keepVertex, connectToCollapsedStableVertex);
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

    std::vector<std::pair<uint32_t, uint32_t>> particleToDecayVertexPairs;
    std::vector<std::pair<uint32_t, uint32_t>> particleToProductionVertexPairs;
    std::vector<std::pair<uint32_t, uint32_t>> vertexToOutgoingParticlePairs;
    std::vector<std::pair<uint32_t, uint32_t>> vertexToIncomingParticlePairs;

    for (uint32_t oldVertex = 0; oldVertex < nVertices; ++oldVertex) {
      const int32_t newVertex = oldVertexToNew[oldVertex];
      if (newVertex < 0)
        continue;

      for (uint32_t particleId : input.incomingParticles(oldVertex)) {
        if (particleId >= nParticles)
          continue;

        particleToDecayVertexPairs.emplace_back(particleId, static_cast<uint32_t>(newVertex));
        vertexToIncomingParticlePairs.emplace_back(static_cast<uint32_t>(newVertex), particleId);
      }

      for (uint32_t particleId : input.outgoingParticles(oldVertex)) {
        if (particleId >= nParticles)
          continue;

        vertexToOutgoingParticlePairs.emplace_back(static_cast<uint32_t>(newVertex), particleId);
        particleToProductionVertexPairs.emplace_back(particleId, static_cast<uint32_t>(newVertex));
      }
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

    auto getNewVertex = [&](uint32_t oldVertex) -> uint32_t {
      if (oldVertexToNew[oldVertex] >= 0)
        return static_cast<uint32_t>(oldVertexToNew[oldVertex]);

      const int rep = vertexDSU.find(static_cast<int>(oldVertex));
      auto inserted = vertexRepToNew.emplace(rep, static_cast<uint32_t>(output.vertices.size()));

      if (inserted.second) {
        output.vertices.push_back(input.vertices[oldVertex]);
      }

      oldVertexToNew[oldVertex] = static_cast<int32_t>(inserted.first->second);
      return inserted.first->second;
    };

    std::vector<std::pair<uint32_t, uint32_t>> particleToDecayVertexPairs;
    std::vector<std::pair<uint32_t, uint32_t>> particleToProductionVertexPairs;
    std::vector<std::pair<uint32_t, uint32_t>> vertexToOutgoingParticlePairs;
    std::vector<std::pair<uint32_t, uint32_t>> vertexToIncomingParticlePairs;

    for (uint32_t oldVertex = 0; oldVertex < nVertices; ++oldVertex) {
      bool hasVisibleIncoming = false;
      bool hasVisibleOutgoing = false;

      for (uint32_t oldParticle : input.incomingParticles(oldVertex)) {
        if (oldParticle < nParticles && oldParticleToNew[oldParticle] >= 0) {
          hasVisibleIncoming = true;
          break;
        }
      }

      for (uint32_t oldParticle : input.outgoingParticles(oldVertex)) {
        if (oldParticle < nParticles && oldParticleToNew[oldParticle] >= 0) {
          hasVisibleOutgoing = true;
          break;
        }
      }

      if (!hasVisibleIncoming && !hasVisibleOutgoing)
        continue;

      const uint32_t newVertex = getNewVertex(oldVertex);

      for (uint32_t oldParticle : input.incomingParticles(oldVertex)) {
        if (oldParticle >= nParticles)
          continue;

        const int32_t newParticle = oldParticleToNew[oldParticle];
        if (newParticle < 0)
          continue;

        particleToDecayVertexPairs.emplace_back(static_cast<uint32_t>(newParticle), newVertex);
        vertexToIncomingParticlePairs.emplace_back(newVertex, static_cast<uint32_t>(newParticle));
      }

      for (uint32_t oldParticle : input.outgoingParticles(oldVertex)) {
        if (oldParticle >= nParticles)
          continue;

        const int32_t newParticle = oldParticleToNew[oldParticle];
        if (newParticle < 0)
          continue;

        vertexToOutgoingParticlePairs.emplace_back(newVertex, static_cast<uint32_t>(newParticle));
        particleToProductionVertexPairs.emplace_back(static_cast<uint32_t>(newParticle), newVertex);
      }
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
            "If non-empty, keep particles with these exact PDG ids, keep seedParentDepth generations above them, "
            "then keep the full downstream subgraph. Stable GEN particles outside the selected subgraph are kept "
            "and attached to one artificial collapsed vertex.");

    desc.add<uint32_t>("seedParentDepth", 0)
        ->setComment("Number of parent generations to keep above each seed particle before keeping downstream.");

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

    desc.add<double>("genSimVertexPositionTolerance", 1e-6)
        ->setComment("Absolute tolerance applied independently to x, y, z, and t when merging GEN and SIM vertices.");

    return desc;
  }

  LogicalGraphPostProcessingConfig TruthLogicalGraphPostProcessor::configFromPSet(edm::ParameterSet const& pset) {
    LogicalGraphPostProcessingConfig config;

    config.collapseIntermediateGenParticles = pset.getParameter<bool>("collapseIntermediateGenParticles");
    config.seedPdgIds = pset.getParameter<std::vector<int32_t>>("seedPdgIds");
    config.seedParentDepth = pset.getParameter<uint32_t>("seedParentDepth");
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

    input = filterGraphBySeedPdgIds(input, config_);

    if (config_.mergeGenSimVerticesByPosition) {
      input = mergeGenSimVerticesByPosition(input, config_.genSimVertexPositionTolerance);
    }

    if (!config_.ignoredPdgIds.empty() || !config_.ignoredParticleIds.empty()) {
      input = collapseIgnoredParticles(input, config_.ignoredPdgIds, config_.ignoredParticleIds);
    }

    return input;
  }

}  // namespace truth
