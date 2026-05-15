#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#include "FWCore/Utilities/interface/Exception.h"
#include "PhysicsTools/TruthInfo/interface/Graph.h"
#include "PhysicsTools/TruthInfo/interface/TruthLogicalGraphPostProcessor.h"

namespace {

  void buildCSR(uint32_t nSources,
                std::vector<std::pair<uint32_t, uint32_t>>& pairs,
                std::vector<uint32_t>& offsets,
                std::vector<uint32_t>& flat) {
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

    offsets.assign(nSources + 1, 0);

    for (auto const& pair : pairs) {
      CPPUNIT_ASSERT(pair.first < nSources);
      ++offsets[pair.first + 1];
    }

    for (uint32_t i = 1; i <= nSources; ++i) {
      offsets[i] += offsets[i - 1];
    }

    flat.assign(pairs.size(), 0);
    auto cursor = offsets;

    for (auto const& pair : pairs) {
      flat[cursor[pair.first]++] = pair.second;
    }
  }

  struct GraphBuilder {
    explicit GraphBuilder(uint32_t nParticles, uint32_t nVertices) {
      graph.particles.resize(nParticles);
      graph.vertices.resize(nVertices);
    }

    void setGenParticle(uint32_t particleId, int32_t pdgId, int16_t status, int32_t genNode) {
      CPPUNIT_ASSERT(particleId < graph.nParticles());

      auto& particle = graph.particles[particleId];
      particle.genNode = genNode;
      particle.simNode = -1;
      particle.pdgId = pdgId;
      particle.status = status;
      particle.statusFlags = 0;
      particle.eventId = 0;
      particle.genEvent = 0;
    }

    void setSimParticle(uint32_t particleId, int32_t pdgId, int32_t simNode) {
      CPPUNIT_ASSERT(particleId < graph.nParticles());

      auto& particle = graph.particles[particleId];
      particle.genNode = -1;
      particle.simNode = simNode;
      particle.pdgId = pdgId;
      particle.status = 0;
      particle.statusFlags = 0;
      particle.eventId = 1;
      particle.genEvent = -1;
    }

    void setGenSimParticle(uint32_t particleId, int32_t pdgId, int16_t status, int32_t genNode, int32_t simNode) {
      CPPUNIT_ASSERT(particleId < graph.nParticles());

      auto& particle = graph.particles[particleId];
      particle.genNode = genNode;
      particle.simNode = simNode;
      particle.pdgId = pdgId;
      particle.status = status;
      particle.statusFlags = 0;
      particle.eventId = 1;
      particle.genEvent = 0;
    }

    void setGenVertex(uint32_t vertexId, int32_t genNode) {
      CPPUNIT_ASSERT(vertexId < graph.nVertices());

      auto& vertex = graph.vertices[vertexId];
      vertex.genNode = genNode;
      vertex.simNode = -1;
      vertex.eventId = 0;
      vertex.genEvent = 0;
    }

    void setSimVertex(uint32_t vertexId, int32_t simNode) {
      CPPUNIT_ASSERT(vertexId < graph.nVertices());

      auto& vertex = graph.vertices[vertexId];
      vertex.genNode = -1;
      vertex.simNode = simNode;
      vertex.eventId = 1;
      vertex.genEvent = -1;
    }

    void setGenSimVertex(uint32_t vertexId, int32_t genNode, int32_t simNode) {
      CPPUNIT_ASSERT(vertexId < graph.nVertices());

      auto& vertex = graph.vertices[vertexId];
      vertex.genNode = genNode;
      vertex.simNode = simNode;
      vertex.eventId = 1;
      vertex.genEvent = 0;
    }

    void addDecay(uint32_t particleId, uint32_t vertexId) {
      CPPUNIT_ASSERT(particleId < graph.nParticles());
      CPPUNIT_ASSERT(vertexId < graph.nVertices());

      particleToDecayVertexPairs.emplace_back(particleId, vertexId);
      vertexToIncomingParticlePairs.emplace_back(vertexId, particleId);
    }

    void addProduction(uint32_t vertexId, uint32_t particleId) {
      CPPUNIT_ASSERT(vertexId < graph.nVertices());
      CPPUNIT_ASSERT(particleId < graph.nParticles());

      vertexToOutgoingParticlePairs.emplace_back(vertexId, particleId);
      particleToProductionVertexPairs.emplace_back(particleId, vertexId);
    }

    truth::Graph finish() {
      buildCSR(graph.nParticles(),
               particleToDecayVertexPairs,
               graph.particleToDecayVertexOffsets,
               graph.particleToDecayVertices);

      buildCSR(graph.nParticles(),
               particleToProductionVertexPairs,
               graph.particleToProductionVertexOffsets,
               graph.particleToProductionVertices);

      buildCSR(graph.nVertices(),
               vertexToOutgoingParticlePairs,
               graph.vertexToOutgoingParticleOffsets,
               graph.vertexToOutgoingParticles);

      buildCSR(graph.nVertices(),
               vertexToIncomingParticlePairs,
               graph.vertexToIncomingParticleOffsets,
               graph.vertexToIncomingParticles);

      CPPUNIT_ASSERT(graph.isConsistent());

      return graph;
    }

    truth::Graph graph;

    std::vector<std::pair<uint32_t, uint32_t>> particleToDecayVertexPairs;
    std::vector<std::pair<uint32_t, uint32_t>> particleToProductionVertexPairs;
    std::vector<std::pair<uint32_t, uint32_t>> vertexToOutgoingParticlePairs;
    std::vector<std::pair<uint32_t, uint32_t>> vertexToIncomingParticlePairs;
  };

  uint32_t countParticlesWithPdgId(truth::Graph const& graph, int32_t pdgId) {
    uint32_t count = 0;

    for (auto const& particle : graph.particles) {
      if (particle.pdgId == pdgId)
        ++count;
    }

    return count;
  }

  uint32_t countStableGenParticles(truth::Graph const& graph) {
    uint32_t count = 0;

    for (auto const& particle : graph.particles) {
      if (particle.hasGen() && particle.status == 1)
        ++count;
    }

    return count;
  }

  bool hasGenSimParticleWithPdgId(truth::Graph const& graph, int32_t pdgId) {
    return std::any_of(graph.particles.begin(), graph.particles.end(), [pdgId](auto const& particle) {
      return particle.pdgId == pdgId && particle.hasGen() && particle.hasSim();
    });
  }

  bool hasArtificialVertex(truth::Graph const& graph) {
    return std::any_of(graph.vertices.begin(), graph.vertices.end(), [](auto const& vertex) {
      return !vertex.hasGen() && !vertex.hasSim();
    });
  }

  uint32_t artificialVertexId(truth::Graph const& graph) {
    for (uint32_t i = 0; i < graph.nVertices(); ++i) {
      auto const& vertex = graph.vertices[i];

      if (!vertex.hasGen() && !vertex.hasSim())
        return i;
    }

    CPPUNIT_ASSERT(false);
    return 0;
  }

  uint32_t findParticleWithPdgId(truth::Graph const& graph, int32_t pdgId) {
    for (uint32_t i = 0; i < graph.nParticles(); ++i) {
      if (graph.particles[i].pdgId == pdgId)
        return i;
    }

    CPPUNIT_ASSERT(false);
    return 0;
  }

  truth::LogicalGraphPostProcessingConfig defaultConfig() {
    truth::LogicalGraphPostProcessingConfig config;
    config.collapseIntermediateGenParticles = false;
    config.seedPdgIds = {};
    config.seedParentDepth = 0;
    config.ignoredPdgIds = {};
    config.ignoredParticleIds = {};
    return config;
  }

  truth::Graph runPostProcessing(truth::Graph graph, truth::LogicalGraphPostProcessingConfig const& config) {
    truth::TruthLogicalGraphPostProcessor processor(config);
    return processor.process(std::move(graph));
  }

}  // namespace

class TestTruthLogicalGraphPostProcessor : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestTruthLogicalGraphPostProcessor);
  CPPUNIT_TEST(testStatusOneGenParticlesAreNeverCollapsed);
  CPPUNIT_TEST(testStableGenSimParticlesSurviveIntermediateCollapse);
  CPPUNIT_TEST(testSeedCutKeepsUnrelatedStableGenSimParticlesThroughArtificialVertex);
  CPPUNIT_TEST(testDagClosureKeepsAllParentsOfKeptVertices);
  CPPUNIT_TEST(testIgnoredParticlesAreCollapsedAway);
  CPPUNIT_TEST(testSeedCutWithIgnoredParticles);
  CPPUNIT_TEST(testIgnoredParticleIdsAreCollapsedAway);
  CPPUNIT_TEST_SUITE_END();

public:
  void testStatusOneGenParticlesAreNeverCollapsed();
  void testStableGenSimParticlesSurviveIntermediateCollapse();
  void testSeedCutKeepsUnrelatedStableGenSimParticlesThroughArtificialVertex();
  void testDagClosureKeepsAllParentsOfKeptVertices();
  void testIgnoredParticlesAreCollapsedAway();
  void testSeedCutWithIgnoredParticles();
  void testIgnoredParticleIdsAreCollapsedAway();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestTruthLogicalGraphPostProcessor);

void TestTruthLogicalGraphPostProcessor::testStatusOneGenParticlesAreNeverCollapsed() {
  try {
    GraphBuilder builder(2, 1);

    // This topology is intentionally unphysical: a status-1 GEN particle has a
    // decay vertex to another same-PDG status-1 particle. The postprocessor must
    // still never collapse a status-1 GEN particle.
    builder.setGenParticle(0, 22, 1, 100);
    builder.setGenParticle(1, 22, 1, 101);
    builder.setGenVertex(0, 200);

    builder.addDecay(0, 0);
    builder.addProduction(0, 1);

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.collapseIntermediateGenParticles = true;

    auto output = runPostProcessing(std::move(graph), config);

    CPPUNIT_ASSERT(output.isConsistent());
    CPPUNIT_ASSERT_EQUAL(uint32_t(2), output.nParticles());
    CPPUNIT_ASSERT_EQUAL(uint32_t(2), countParticlesWithPdgId(output, 22));
    CPPUNIT_ASSERT_EQUAL(uint32_t(2), countStableGenParticles(output));
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testStableGenSimParticlesSurviveIntermediateCollapse() {
  try {
    GraphBuilder builder(3, 2);

    // gamma(status 2) -> gamma(status 1, GEN+SIM)
    // e-(status 1, GEN+SIM) is an independent stable final-state particle.
    builder.setGenParticle(0, 22, 2, 100);
    builder.setGenSimParticle(1, 22, 1, 101, 1001);
    builder.setGenSimParticle(2, 11, 1, 102, 1002);

    builder.setGenVertex(0, 200);
    builder.setGenSimVertex(1, 201, 2001);

    builder.addDecay(0, 0);
    builder.addProduction(0, 1);
    builder.addProduction(1, 2);

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.collapseIntermediateGenParticles = true;

    auto output = runPostProcessing(std::move(graph), config);

    CPPUNIT_ASSERT(output.isConsistent());

    // The intermediate gamma can collapse into the final stable gamma, but the
    // status-1 GEN+SIM gamma must remain materialized.
    CPPUNIT_ASSERT_EQUAL(uint32_t(2), output.nParticles());
    CPPUNIT_ASSERT(hasGenSimParticleWithPdgId(output, 22));
    CPPUNIT_ASSERT(hasGenSimParticleWithPdgId(output, 11));
    CPPUNIT_ASSERT_EQUAL(uint32_t(2), countStableGenParticles(output));
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testSeedCutKeepsUnrelatedStableGenSimParticlesThroughArtificialVertex() {
  try {
    GraphBuilder builder(4, 3);

    // Interesting branch:
    //   Z -> gamma(status 1, GEN+SIM)
    //
    // Unrelated stable final state:
    //   artificial filtering must keep e-(status 1, GEN+SIM), but attach it to
    //   one artificial vertex instead of keeping its unrelated production chain.
    builder.setGenParticle(0, 23, 2, 100);
    builder.setGenSimParticle(1, 22, 1, 101, 1001);
    builder.setGenParticle(2, 999, 2, 102);
    builder.setGenSimParticle(3, 11, 1, 103, 1003);

    builder.setGenSimVertex(0, 200, 2000);
    builder.setGenVertex(1, 201);
    builder.setGenSimVertex(2, 202, 2002);

    builder.addDecay(0, 0);
    builder.addProduction(0, 1);

    builder.addDecay(2, 1);
    builder.addProduction(1, 3);

    // A second production vertex for the stable electron, to make sure the
    // filtered graph does not keep unrelated upstream structure.
    builder.addProduction(2, 3);

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.seedPdgIds = {22};
    config.seedParentDepth = 1;

    auto output = runPostProcessing(std::move(graph), config);

    CPPUNIT_ASSERT(output.isConsistent());

    CPPUNIT_ASSERT(hasGenSimParticleWithPdgId(output, 22));
    CPPUNIT_ASSERT(hasGenSimParticleWithPdgId(output, 11));
    CPPUNIT_ASSERT(hasArtificialVertex(output));

    const uint32_t electron = findParticleWithPdgId(output, 11);
    const auto productionVertices = output.productionVertices(electron);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), productionVertices.size());

    const uint32_t collapsedVertex = artificialVertexId(output);
    CPPUNIT_ASSERT_EQUAL(collapsedVertex, productionVertices.front());

    const auto artificialOutgoing = output.outgoingParticles(collapsedVertex);
    CPPUNIT_ASSERT(std::find(artificialOutgoing.begin(), artificialOutgoing.end(), electron) !=
                   artificialOutgoing.end());

    const auto artificialIncoming = output.incomingParticles(collapsedVertex);
    CPPUNIT_ASSERT(artificialIncoming.empty());
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testDagClosureKeepsAllParentsOfKeptVertices() {
  try {
    GraphBuilder builder(5, 3);

    // DAG topology:
    //
    //   H -> v0 -> pi0
    //   Z --------^
    //   pi0 -> v1 -> gamma
    //   e- stable, unrelated
    //
    // The seed is H. Keeping downstream from H keeps v0. Since v0 also has Z as
    // an incoming parent, the postprocessor must include Z and the upstream path
    // to Z instead of showing v0 as if it had only H as parent.
    builder.setGenParticle(0, 25, 2, 100);
    builder.setGenParticle(1, 23, 2, 101);
    builder.setGenParticle(2, 111, 2, 102);
    builder.setGenSimParticle(3, 22, 1, 103, 1003);
    builder.setGenSimParticle(4, 11, 1, 104, 1004);

    builder.setGenVertex(0, 200);
    builder.setGenVertex(1, 201);
    builder.setGenSimVertex(2, 202, 2002);

    builder.addDecay(0, 0);
    builder.addDecay(1, 0);
    builder.addProduction(0, 2);

    builder.addDecay(2, 1);
    builder.addProduction(1, 3);

    builder.addProduction(2, 4);

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.seedPdgIds = {25};
    config.seedParentDepth = 0;

    auto output = runPostProcessing(std::move(graph), config);

    CPPUNIT_ASSERT(output.isConsistent());

    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 25));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 23));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 111));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 22));

    // The unrelated stable electron is still kept, but via the artificial vertex.
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 11));
    CPPUNIT_ASSERT(hasArtificialVertex(output));

    const uint32_t pi0 = findParticleWithPdgId(output, 111);
    const auto pi0ProductionVertices = output.productionVertices(pi0);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), pi0ProductionVertices.size());

    const auto incoming = output.incomingParticles(pi0ProductionVertices.front());

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), incoming.size());

    std::vector<int32_t> incomingPdgIds;
    incomingPdgIds.reserve(incoming.size());

    for (uint32_t parent : incoming) {
      incomingPdgIds.push_back(output.particles[parent].pdgId);
    }

    std::sort(incomingPdgIds.begin(), incomingPdgIds.end());

    CPPUNIT_ASSERT_EQUAL(int32_t(23), incomingPdgIds[0]);
    CPPUNIT_ASSERT_EQUAL(int32_t(25), incomingPdgIds[1]);
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testIgnoredParticlesAreCollapsedAway() {
  try {
    GraphBuilder builder(3, 2);

    // Z -> gamma -> e-
    //
    // If gamma is ignored, it should disappear and the two vertices around it
    // should be merged, preserving a navigable Z -> e- connection.
    builder.setGenParticle(0, 23, 2, 100);
    builder.setGenParticle(1, 22, 2, 101);
    builder.setGenSimParticle(2, 11, 1, 102, 1002);

    builder.setGenVertex(0, 200);
    builder.setGenSimVertex(1, 201, 2001);

    builder.addDecay(0, 0);
    builder.addProduction(0, 1);

    builder.addDecay(1, 1);
    builder.addProduction(1, 2);

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.ignoredPdgIds = {22};

    auto output = runPostProcessing(std::move(graph), config);

    CPPUNIT_ASSERT(output.isConsistent());

    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 22));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 23));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 11));

    const uint32_t z = findParticleWithPdgId(output, 23);
    const uint32_t electron = findParticleWithPdgId(output, 11);

    const auto decayVertices = output.decayVertices(z);
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), decayVertices.size());

    const auto outgoing = output.outgoingParticles(decayVertices.front());
    CPPUNIT_ASSERT(std::find(outgoing.begin(), outgoing.end(), electron) != outgoing.end());
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testSeedCutWithIgnoredParticles() {
  try {
    GraphBuilder builder(6, 4);

    // Interesting branch:
    //   H -> pi0 -> gamma(status 1, GEN+SIM)
    //
    // Extra parent on the kept pi0 production vertex:
    //   Z --------^
    //
    // Unrelated stable final-state e- is kept through the artificial vertex.
    //
    // Then ignoredPdgIds removes pi0, merging H/Z directly to gamma.
    builder.setGenParticle(0, 25, 2, 100);
    builder.setGenParticle(1, 23, 2, 101);
    builder.setGenParticle(2, 111, 2, 102);
    builder.setGenSimParticle(3, 22, 1, 103, 1003);
    builder.setGenParticle(4, 999, 2, 104);
    builder.setGenSimParticle(5, 11, 1, 105, 1005);

    builder.setGenVertex(0, 200);
    builder.setGenSimVertex(1, 201, 2001);
    builder.setGenVertex(2, 202);
    builder.setGenSimVertex(3, 203, 2003);

    builder.addDecay(0, 0);
    builder.addDecay(1, 0);
    builder.addProduction(0, 2);

    builder.addDecay(2, 1);
    builder.addProduction(1, 3);

    builder.addDecay(4, 2);
    builder.addProduction(2, 5);

    builder.addProduction(3, 5);

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.seedPdgIds = {25};
    config.seedParentDepth = 0;
    config.ignoredPdgIds = {111};

    auto output = runPostProcessing(std::move(graph), config);

    CPPUNIT_ASSERT(output.isConsistent());

    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 25));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 23));
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 111));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 22));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 11));

    CPPUNIT_ASSERT(hasGenSimParticleWithPdgId(output, 22));
    CPPUNIT_ASSERT(hasGenSimParticleWithPdgId(output, 11));
    CPPUNIT_ASSERT(hasArtificialVertex(output));

    const uint32_t gamma = findParticleWithPdgId(output, 22);
    const auto gammaProductionVertices = output.productionVertices(gamma);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), gammaProductionVertices.size());

    const auto incoming = output.incomingParticles(gammaProductionVertices.front());

    std::vector<int32_t> incomingPdgIds;
    incomingPdgIds.reserve(incoming.size());

    for (uint32_t parent : incoming) {
      incomingPdgIds.push_back(output.particles[parent].pdgId);
    }

    std::sort(incomingPdgIds.begin(), incomingPdgIds.end());

    CPPUNIT_ASSERT_EQUAL(std::size_t(2), incomingPdgIds.size());
    CPPUNIT_ASSERT_EQUAL(int32_t(23), incomingPdgIds[0]);
    CPPUNIT_ASSERT_EQUAL(int32_t(25), incomingPdgIds[1]);

    const uint32_t electron = findParticleWithPdgId(output, 11);
    const auto electronProductionVertices = output.productionVertices(electron);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), electronProductionVertices.size());
    CPPUNIT_ASSERT_EQUAL(artificialVertexId(output), electronProductionVertices.front());
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testIgnoredParticleIdsAreCollapsedAway() {
  try {
    GraphBuilder builder(4, 3);

    // Z -> a -> gamma -> e-
    //
    // Only particle id 2 is ignored. This verifies that ignoredParticleIds is
    // independent from PDG id matching: the status-1 gamma is removed because its
    // logical id is explicitly listed, not because all photons are ignored.
    builder.setGenParticle(0, 23, 2, 100);
    builder.setGenParticle(1, 36, 2, 101);
    builder.setGenSimParticle(2, 22, 1, 102, 1002);
    builder.setGenSimParticle(3, 11, 1, 103, 1003);

    builder.setGenVertex(0, 200);
    builder.setGenSimVertex(1, 201, 2001);
    builder.setGenSimVertex(2, 202, 2002);

    builder.addDecay(0, 0);
    builder.addProduction(0, 1);

    builder.addDecay(1, 1);
    builder.addProduction(1, 2);

    builder.addDecay(2, 2);
    builder.addProduction(2, 3);

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.ignoredParticleIds = {2};

    auto output = runPostProcessing(std::move(graph), config);

    CPPUNIT_ASSERT(output.isConsistent());

    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 23));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 36));
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 22));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 11));

    const uint32_t a = findParticleWithPdgId(output, 36);
    const uint32_t electron = findParticleWithPdgId(output, 11);

    const auto decayVertices = output.decayVertices(a);
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), decayVertices.size());

    const auto outgoing = output.outgoingParticles(decayVertices.front());
    CPPUNIT_ASSERT(std::find(outgoing.begin(), outgoing.end(), electron) != outgoing.end());
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}
