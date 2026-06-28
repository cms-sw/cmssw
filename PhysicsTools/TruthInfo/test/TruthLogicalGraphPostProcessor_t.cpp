// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#include "FWCore/Utilities/interface/Exception.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "PhysicsTools/TruthInfo/interface/TruthLogicalGraphPostProcessor.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

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
      graph.particles().resize(nParticles);
      graph.vertices().resize(nVertices);
    }

    void setGenParticle(uint32_t particleId, int32_t pdgId, int16_t status, int32_t genNode) {
      CPPUNIT_ASSERT(particleId < graph.nParticles());

      auto& particle = graph.particles()[particleId];
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

      auto& particle = graph.particles()[particleId];
      particle.genNode = -1;
      particle.simNode = simNode;
      particle.pdgId = pdgId;
      particle.status = 0;
      particle.statusFlags = 0;
      particle.eventId = 0;  // signal interaction: EncodedEventId(0, 0) packs to 0
      particle.genEvent = -1;
    }

    void setGenSimParticle(uint32_t particleId, int32_t pdgId, int16_t status, int32_t genNode, int32_t simNode) {
      CPPUNIT_ASSERT(particleId < graph.nParticles());

      auto& particle = graph.particles()[particleId];
      particle.genNode = genNode;
      particle.simNode = simNode;
      particle.pdgId = pdgId;
      particle.status = status;
      particle.statusFlags = 0;
      particle.eventId = 0;  // signal interaction: EncodedEventId(0, 0) packs to 0
      particle.genEvent = 0;
    }

    void setGenVertex(uint32_t vertexId, int32_t genNode) {
      CPPUNIT_ASSERT(vertexId < graph.nVertices());

      auto& vertex = graph.vertices()[vertexId];
      vertex.genNode = genNode;
      vertex.simNode = -1;
      vertex.eventId = 0;
      vertex.genEvent = 0;
    }

    void setSimVertex(uint32_t vertexId, int32_t simNode) {
      CPPUNIT_ASSERT(vertexId < graph.nVertices());

      auto& vertex = graph.vertices()[vertexId];
      vertex.genNode = -1;
      vertex.simNode = simNode;
      vertex.eventId = 0;  // signal interaction
      vertex.genEvent = -1;
    }

    void setGenSimVertex(uint32_t vertexId, int32_t genNode, int32_t simNode) {
      CPPUNIT_ASSERT(vertexId < graph.nVertices());

      auto& vertex = graph.vertices()[vertexId];
      vertex.genNode = genNode;
      vertex.simNode = simNode;
      vertex.eventId = 0;  // signal interaction
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
               graph.particleToDecayVertexOffsets(),
               graph.particleToDecayVertices());

      buildCSR(graph.nParticles(),
               particleToProductionVertexPairs,
               graph.particleToProductionVertexOffsets(),
               graph.particleToProductionVertices());

      buildCSR(graph.nVertices(),
               vertexToOutgoingParticlePairs,
               graph.vertexToOutgoingParticleOffsets(),
               graph.vertexToOutgoingParticles());

      buildCSR(graph.nVertices(),
               vertexToIncomingParticlePairs,
               graph.vertexToIncomingParticleOffsets(),
               graph.vertexToIncomingParticles());

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

    for (auto const& particle : graph.particles()) {
      if (particle.pdgId == pdgId)
        ++count;
    }

    return count;
  }

  uint32_t countStableGenParticles(truth::Graph const& graph) {
    uint32_t count = 0;

    for (auto const& particle : graph.particles()) {
      if (particle.hasGen() && particle.status == 1)
        ++count;
    }

    return count;
  }

  bool hasGenSimParticleWithPdgId(truth::Graph const& graph, int32_t pdgId) {
    return std::any_of(graph.particles().begin(), graph.particles().end(), [pdgId](auto const& particle) {
      return particle.pdgId == pdgId && particle.hasGen() && particle.hasSim();
    });
  }

  bool hasArtificialVertex(truth::Graph const& graph) {
    return std::any_of(graph.vertices().begin(), graph.vertices().end(), [](auto const& vertex) {
      return !vertex.hasGen() && !vertex.hasSim();
    });
  }

  // The artificial *sub*-vertex a particle attaches to (Upstream or
  // UnderlyingEvent), i.e. skipping the per-interaction Interaction root that
  // those sub-vertices descend from.
  uint32_t artificialVertexId(truth::Graph const& graph) {
    for (uint32_t i = 0; i < graph.nVertices(); ++i) {
      auto const& vertex = graph.vertices()[i];

      if (vertex.isArtificial() && vertex.vertexRole() != truth::VertexRole::Interaction)
        return i;
    }

    CPPUNIT_ASSERT(false);
    return 0;
  }

  uint32_t findParticleWithPdgId(truth::Graph const& graph, int32_t pdgId) {
    for (uint32_t i = 0; i < graph.nParticles(); ++i) {
      if (graph.particles()[i].pdgId == pdgId)
        return i;
    }

    CPPUNIT_ASSERT(false);
    return 0;
  }

  uint32_t findVertexWithRole(truth::Graph const& graph, truth::VertexRole role) {
    for (uint32_t i = 0; i < graph.nVertices(); ++i) {
      if (graph.vertices()[i].vertexRole() == role)
        return i;
    }

    CPPUNIT_ASSERT(false);
    return 0;
  }

  uint32_t countArtificialVerticesWithRole(truth::Graph const& graph, truth::VertexRole role) {
    uint32_t count = 0;
    for (auto const& vertex : graph.vertices()) {
      if (vertex.isArtificial() && vertex.vertexRole() == role)
        ++count;
    }
    return count;
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

  truth::Graph runPostProcessing(truth::Graph graph,
                                 truth::LogicalGraphPostProcessingConfig const& config,
                                 std::vector<uint8_t> const& particleDirectHit) {
    truth::TruthLogicalGraphPostProcessor processor(config);
    return processor.process(std::move(graph), particleDirectHit);
  }

}  // namespace

class TestTruthLogicalGraphPostProcessor : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestTruthLogicalGraphPostProcessor);
  CPPUNIT_TEST(testStatusOneGenParticlesAreNeverCollapsed);
  CPPUNIT_TEST(testStableGenSimParticlesSurviveIntermediateCollapse);
  CPPUNIT_TEST(testSeedCutKeepsUnrelatedStableGenSimParticlesThroughArtificialVertex);
  CPPUNIT_TEST(testSeedCutHidesUnselectedParentsOfKeptVertices);
  CPPUNIT_TEST(testIgnoredParticlesAreCollapsedAway);
  CPPUNIT_TEST(testSeedCutWithIgnoredParticles);
  CPPUNIT_TEST(testIgnoredParticleIdsAreCollapsedAway);
  CPPUNIT_TEST(testSeedRootIsMostUpstreamThroughRadiatingCopyChain);
  CPPUNIT_TEST(testSeedParentDepthKeepsAncestorContextOnly);
  CPPUNIT_TEST(testKeepProductionSiblingsKeepsHardCoProducts);
  CPPUNIT_TEST(testSeedWithDecayGroupKeepsOnlyMatchingDecays);
  CPPUNIT_TEST(testZToTauTauDoesNotMatchMuonDecayGroup);
  CPPUNIT_TEST(testDecayGroupFallbackWhenSeedAbsent);
  CPPUNIT_TEST(testSeedPdgIdZeroKeepsFullGraphForDebugging);
  CPPUNIT_TEST(testArtificialSourceRolesAndProvenance);
  CPPUNIT_TEST(testKeepStableSpectatorsFalseDropsSpectators);
  CPPUNIT_TEST(testSeedHadronFlavorSelectsBHadron);
  CPPUNIT_TEST(testJetOriginLowestCommonAncestor);
  CPPUNIT_TEST(testHitlessSimSubgraphsAreDropped);
  CPPUNIT_TEST(testAttachSelectionSourcesFalseRootsSeedsDirectly);
  CPPUNIT_TEST(testEventIdKeyingSplitsInteractions);
  CPPUNIT_TEST(testSignalOnlyAndBunchCrossingFilterDropPileup);
  CPPUNIT_TEST_SUITE_END();

public:
  void testStatusOneGenParticlesAreNeverCollapsed();
  void testStableGenSimParticlesSurviveIntermediateCollapse();
  void testSeedCutKeepsUnrelatedStableGenSimParticlesThroughArtificialVertex();
  void testSeedCutHidesUnselectedParentsOfKeptVertices();
  void testIgnoredParticlesAreCollapsedAway();
  void testSeedCutWithIgnoredParticles();
  void testIgnoredParticleIdsAreCollapsedAway();
  void testSeedRootIsMostUpstreamThroughRadiatingCopyChain();
  void testSeedParentDepthKeepsAncestorContextOnly();
  void testKeepProductionSiblingsKeepsHardCoProducts();
  void testSeedWithDecayGroupKeepsOnlyMatchingDecays();
  void testZToTauTauDoesNotMatchMuonDecayGroup();
  void testDecayGroupFallbackWhenSeedAbsent();
  void testSeedPdgIdZeroKeepsFullGraphForDebugging();
  void testArtificialSourceRolesAndProvenance();
  void testKeepStableSpectatorsFalseDropsSpectators();
  void testSeedHadronFlavorSelectsBHadron();
  void testJetOriginLowestCommonAncestor();
  void testHitlessSimSubgraphsAreDropped();
  void testAttachSelectionSourcesFalseRootsSeedsDirectly();
  void testEventIdKeyingSplitsInteractions();
  void testSignalOnlyAndBunchCrossingFilterDropPileup();
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

    // The sub-vertex is not a source: it descends from the per-interaction
    // Interaction vertex through one artificial connector particle.
    const auto artificialIncoming = output.incomingParticles(collapsedVertex);
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), artificialIncoming.size());
    const auto connectorProduction = output.productionVertices(artificialIncoming.front());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), connectorProduction.size());
    CPPUNIT_ASSERT(output.vertices()[connectorProduction.front()].vertexRole() == truth::VertexRole::Interaction);
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testSeedCutHidesUnselectedParentsOfKeptVertices() {
  try {
    GraphBuilder builder(5, 3);

    // DAG topology:
    //
    //   H -> v0 -> pi0
    //   Z --------^
    //   pi0 -> v1 -> gamma
    //   e- stable, unrelated
    //
    // The seed is H. Keeping downstream from H keeps v0. The unselected Z parent
    // of v0 is not part of the selection and seedParentDepth is 0, so it must be
    // hidden: v0 appears with H as its only incoming particle.
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
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 23));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 111));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 22));

    // The unrelated stable electron is still kept, but via the artificial vertex.
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 11));
    CPPUNIT_ASSERT(hasArtificialVertex(output));

    const uint32_t pi0 = findParticleWithPdgId(output, 111);
    const auto pi0ProductionVertices = output.productionVertices(pi0);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), pi0ProductionVertices.size());

    const auto incoming = output.incomingParticles(pi0ProductionVertices.front());

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), incoming.size());
    CPPUNIT_ASSERT_EQUAL(int32_t(25), output.particles()[incoming.front()].pdgId);

    const uint32_t electron = findParticleWithPdgId(output, 11);
    const auto electronProductionVertices = output.productionVertices(electron);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), electronProductionVertices.size());
    CPPUNIT_ASSERT_EQUAL(artificialVertexId(output), electronProductionVertices.front());
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

void TestTruthLogicalGraphPostProcessor::testSeedRootIsMostUpstreamThroughRadiatingCopyChain() {
  try {
    GraphBuilder builder(6, 3);

    // q -> vq -> Z0
    // Z0 -> v0 -> { Z1, gamma }   (radiating copy chain, survives chain collapse)
    // Z1 -> v1 -> { mu+, mu- }
    //
    // With seedPdgIds = {23} only Z0 is a root: Z1 is a strict descendant of
    // another match. The q parent is outside the selection (depth 0), so Z0 is
    // attached to the artificial source vertex while Z1 keeps its real
    // production vertex.
    builder.setGenParticle(0, 1, 2, 100);
    builder.setGenParticle(1, 23, 2, 101);
    builder.setGenParticle(2, 23, 2, 102);
    builder.setGenSimParticle(3, 22, 1, 103, 1003);
    builder.setGenSimParticle(4, -13, 1, 104, 1004);
    builder.setGenSimParticle(5, 13, 1, 105, 1005);

    builder.setGenVertex(0, 200);
    builder.setGenVertex(1, 201);
    builder.setGenSimVertex(2, 202, 2002);

    builder.addDecay(0, 0);
    builder.addProduction(0, 1);

    builder.addDecay(1, 1);
    builder.addProduction(1, 2);
    builder.addProduction(1, 3);

    builder.addDecay(2, 2);
    builder.addProduction(2, 4);
    builder.addProduction(2, 5);

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.seedPdgIds = {23};
    config.seedParentDepth = 0;

    auto output = runPostProcessing(std::move(graph), config);

    CPPUNIT_ASSERT(output.isConsistent());

    CPPUNIT_ASSERT_EQUAL(uint32_t(2), countParticlesWithPdgId(output, 23));
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 1));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 22));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, -13));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 13));
    CPPUNIT_ASSERT(hasArtificialVertex(output));

    // Exactly one Z (the most upstream root, with its q parent dropped) hangs
    // off the artificial source vertex; the downstream copy keeps its real one.
    const uint32_t artificial = artificialVertexId(output);
    uint32_t nZAttachedToArtificial = 0;

    for (uint32_t particleId = 0; particleId < output.nParticles(); ++particleId) {
      if (output.particles()[particleId].pdgId != 23)
        continue;

      const auto productionVertices = output.productionVertices(particleId);

      if (productionVertices.size() == 1 && productionVertices.front() == artificial)
        ++nZAttachedToArtificial;
    }

    CPPUNIT_ASSERT_EQUAL(uint32_t(1), nZAttachedToArtificial);
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testSeedParentDepthKeepsAncestorContextOnly() {
  try {
    GraphBuilder builder(7, 3);

    // g g -> vp -> { Z, q }
    // Z -> vz -> { mu+, mu- }
    // q -> vq -> { pi+ (stable) }
    //
    // With seedParentDepth = 1 the gluons and vp are kept as context, but the
    // sibling q and its decay chain are not: ancestors no longer pull in their
    // own downstream. The stable pion survives through the artificial vertex.
    builder.setGenParticle(0, 21, 2, 100);
    builder.setGenParticle(1, 21, 2, 101);
    builder.setGenParticle(2, 23, 2, 102);
    builder.setGenParticle(3, 1, 2, 103);
    builder.setGenSimParticle(4, 211, 1, 104, 1004);
    builder.setGenSimParticle(5, -13, 1, 105, 1005);
    builder.setGenSimParticle(6, 13, 1, 106, 1006);

    builder.setGenVertex(0, 200);
    builder.setGenVertex(1, 201);
    builder.setGenSimVertex(2, 202, 2002);

    builder.addDecay(0, 0);
    builder.addDecay(1, 0);
    builder.addProduction(0, 2);
    builder.addProduction(0, 3);

    builder.addDecay(3, 1);
    builder.addProduction(1, 4);

    builder.addDecay(2, 2);
    builder.addProduction(2, 5);
    builder.addProduction(2, 6);

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.seedPdgIds = {23};
    config.seedParentDepth = 1;

    auto output = runPostProcessing(std::move(graph), config);

    CPPUNIT_ASSERT(output.isConsistent());

    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 23));
    CPPUNIT_ASSERT_EQUAL(uint32_t(2), countParticlesWithPdgId(output, 21));
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 1));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 211));
    CPPUNIT_ASSERT(hasArtificialVertex(output));

    const uint32_t z = findParticleWithPdgId(output, 23);
    const auto zProductionVertices = output.productionVertices(z);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), zProductionVertices.size());

    // Both gluons are visible as context above the Z, and the hidden sibling
    // quark leaves the production vertex with the Z as its only outgoing.
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), output.incomingParticles(zProductionVertices.front()).size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), output.outgoingParticles(zProductionVertices.front()).size());

    const uint32_t pion = findParticleWithPdgId(output, 211);
    const auto pionProductionVertices = output.productionVertices(pion);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), pionProductionVertices.size());
    CPPUNIT_ASSERT_EQUAL(artificialVertexId(output), pionProductionVertices.front());
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testKeepProductionSiblingsKeepsHardCoProducts() {
  try {
    GraphBuilder builder(7, 3);

    // g g -> vp -> { Z, q };  Z -> vz -> { mu+, mu- };  q -> vq -> { pi+ (stable) }
    //
    // Same topology as the seedParentDepth test, but seeding on the Z with
    // keepProductionSiblings keeps the recoiling quark q - the Z's sibling at the
    // shared production vertex - and its decay subtree (the pion jet). This is the
    // VBF case in miniature (the quark would be a tagging jet); seedParentDepth
    // alone never reaches it because it is a co-product, not an ancestor.
    builder.setGenParticle(0, 21, 2, 100);
    builder.setGenParticle(1, 21, 2, 101);
    builder.setGenParticle(2, 23, 2, 102);
    builder.setGenParticle(3, 1, 2, 103);
    builder.setGenSimParticle(4, 211, 1, 104, 1004);
    builder.setGenSimParticle(5, -13, 1, 105, 1005);
    builder.setGenSimParticle(6, 13, 1, 106, 1006);

    builder.setGenVertex(0, 200);
    builder.setGenVertex(1, 201);
    builder.setGenSimVertex(2, 202, 2002);

    builder.addDecay(0, 0);
    builder.addDecay(1, 0);
    builder.addProduction(0, 2);
    builder.addProduction(0, 3);

    builder.addDecay(3, 1);
    builder.addProduction(1, 4);

    builder.addDecay(2, 2);
    builder.addProduction(2, 5);
    builder.addProduction(2, 6);

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.seedPdgIds = {23};
    config.seedParentDepth = 0;
    // Drop spectators so the quark/pion can only enter via keepProductionSiblings.
    config.keepStableSpectators = false;
    config.keepProductionSiblings = true;

    auto output = runPostProcessing(std::move(graph), config);

    CPPUNIT_ASSERT(output.isConsistent());

    // The recoiling quark (unlike seedParentDepth, which drops it) and its pion
    // jet are kept.
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 23));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 1));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 211));

    const uint32_t z = findParticleWithPdgId(output, 23);
    const uint32_t quark = findParticleWithPdgId(output, 1);
    const auto zProductionVertices = output.productionVertices(z);
    const auto quarkProductionVertices = output.productionVertices(quark);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), zProductionVertices.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), quarkProductionVertices.size());

    // The Z and the quark share the real hard-scatter production vertex, which now
    // exposes both as outgoing - the recoiling co-product is visible.
    CPPUNIT_ASSERT_EQUAL(zProductionVertices.front(), quarkProductionVertices.front());
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), output.outgoingParticles(zProductionVertices.front()).size());

    // The pion came in through the quark's real decay chain (the jet), so its
    // production vertex is the quark's decay vertex - not an artificial source.
    const uint32_t pion = findParticleWithPdgId(output, 211);
    const auto pionProductionVertices = output.productionVertices(pion);
    const auto quarkDecayVertices = output.decayVertices(quark);
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), pionProductionVertices.size());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), quarkDecayVertices.size());
    CPPUNIT_ASSERT_EQUAL(quarkDecayVertices.front(), pionProductionVertices.front());
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testSeedWithDecayGroupKeepsOnlyMatchingDecays() {
  try {
    GraphBuilder builder(7, 3);

    // Za -> v0 -> { Za', gamma }   (radiating copy)
    // Za' -> v1 -> { mu+, mu- }
    // Zb -> v2 -> { e- }
    //
    // seedPdgIds = {23}, decayPdgIdGroups = {{13, -13}}: the decay match is
    // evaluated at the most upstream root after following the copy chain, so
    // Za (-> mu mu) is kept and Zb (-> e e) is dropped. The stable electrons
    // survive only as spectators on the artificial vertex.
    builder.setGenParticle(0, 23, 2, 100);
    builder.setGenParticle(1, 23, 2, 101);
    builder.setGenParticle(2, 23, 2, 102);
    builder.setGenSimParticle(3, 22, 1, 103, 1003);
    builder.setGenSimParticle(4, -13, 1, 104, 1004);
    builder.setGenSimParticle(5, 13, 1, 105, 1005);
    builder.setGenSimParticle(6, 11, 1, 106, 1006);

    builder.setGenVertex(0, 200);
    builder.setGenSimVertex(1, 201, 2001);
    builder.setGenSimVertex(2, 202, 2002);

    builder.addDecay(0, 0);
    builder.addProduction(0, 1);
    builder.addProduction(0, 3);

    builder.addDecay(1, 1);
    builder.addProduction(1, 4);
    builder.addProduction(1, 5);

    builder.addDecay(2, 2);
    builder.addProduction(2, 6);

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.seedPdgIds = {23};
    config.decayPdgIdGroups = {{13, -13}};
    config.seedParentDepth = 0;

    auto output = runPostProcessing(std::move(graph), config);

    CPPUNIT_ASSERT(output.isConsistent());

    // Both copies of the matching Z chain are kept; the Z -> e e one is gone.
    CPPUNIT_ASSERT_EQUAL(uint32_t(2), countParticlesWithPdgId(output, 23));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, -13));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 13));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 11));
    CPPUNIT_ASSERT(hasArtificialVertex(output));

    const uint32_t muon = findParticleWithPdgId(output, 13);
    const auto muonProductionVertices = output.productionVertices(muon);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), muonProductionVertices.size());
    CPPUNIT_ASSERT(muonProductionVertices.front() != artificialVertexId(output));

    const uint32_t electron = findParticleWithPdgId(output, 11);
    const auto electronProductionVertices = output.productionVertices(electron);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), electronProductionVertices.size());
    CPPUNIT_ASSERT_EQUAL(artificialVertexId(output), electronProductionVertices.front());
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testZToTauTauDoesNotMatchMuonDecayGroup() {
  try {
    GraphBuilder builder(7, 3);

    // Z -> v0 -> { tau+, tau- }
    // tau+ -> v1 -> { mu+, anti-nu }
    // tau- -> v2 -> { mu-, nu }
    //
    // The decay match is local to the (copy-collapsed) decay vertex of the
    // root: the muons from the tau decays must NOT make Z -> tau tau match
    // {13, -13}. Nothing matches, so only stable particles survive, attached
    // to the artificial vertex.
    builder.setGenParticle(0, 23, 2, 100);
    builder.setGenParticle(1, -15, 2, 101);
    builder.setGenParticle(2, 15, 2, 102);
    builder.setGenSimParticle(3, -13, 1, 103, 1003);
    builder.setGenSimParticle(4, 13, 1, 104, 1004);
    builder.setGenParticle(5, -16, 1, 105);
    builder.setGenParticle(6, 16, 1, 106);

    builder.setGenVertex(0, 200);
    builder.setGenSimVertex(1, 201, 2001);
    builder.setGenSimVertex(2, 202, 2002);

    builder.addDecay(0, 0);
    builder.addProduction(0, 1);
    builder.addProduction(0, 2);

    builder.addDecay(1, 1);
    builder.addProduction(1, 3);
    builder.addProduction(1, 5);

    builder.addDecay(2, 2);
    builder.addProduction(2, 4);
    builder.addProduction(2, 6);

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.seedPdgIds = {23};
    config.decayPdgIdGroups = {{13, -13}};

    auto output = runPostProcessing(std::move(graph), config);

    CPPUNIT_ASSERT(output.isConsistent());

    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 23));
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 15));
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, -15));

    // Stable muons and neutrinos survive as spectators only: four real particles
    // plus one artificial connector, under two artificial vertices
    // (Interaction -> UnderlyingEvent).
    CPPUNIT_ASSERT_EQUAL(uint32_t(5), output.nParticles());
    CPPUNIT_ASSERT(hasArtificialVertex(output));
    CPPUNIT_ASSERT_EQUAL(uint32_t(2), output.nVertices());

    const uint32_t muon = findParticleWithPdgId(output, 13);
    const auto muonProductionVertices = output.productionVertices(muon);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), muonProductionVertices.size());
    CPPUNIT_ASSERT_EQUAL(artificialVertexId(output), muonProductionVertices.front());
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testDecayGroupFallbackWhenSeedAbsent() {
  try {
    GraphBuilder builder(5, 2);

    // The generator wrote no explicit Z: vp -> { mu+, mu-, gamma }.
    // Unrelated branch: X -> vx -> { pi+ (stable) }.
    //
    // seedPdgIds = {23} finds nothing, so the decay-pattern fallback selects
    // the mu+ mu- vertex (extra photon allowed). The matched vertex is kept as
    // the common production context; the pion survives via the artificial
    // vertex and X is dropped.
    builder.setGenSimParticle(0, -13, 1, 100, 1000);
    builder.setGenSimParticle(1, 13, 1, 101, 1001);
    builder.setGenSimParticle(2, 22, 1, 102, 1002);
    builder.setGenParticle(3, 999, 2, 103);
    builder.setGenSimParticle(4, 211, 1, 104, 1004);

    builder.setGenVertex(0, 200);
    builder.setGenVertex(1, 201);

    builder.addProduction(0, 0);
    builder.addProduction(0, 1);
    builder.addProduction(0, 2);

    builder.addDecay(3, 1);
    builder.addProduction(1, 4);

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.seedPdgIds = {23};
    config.decayPdgIdGroups = {{13, -13}};

    auto output = runPostProcessing(std::move(graph), config);

    CPPUNIT_ASSERT(output.isConsistent());

    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 999));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, -13));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 13));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 22));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 211));
    CPPUNIT_ASSERT(hasArtificialVertex(output));

    const uint32_t artificial = artificialVertexId(output);

    // The muons hang off their real, kept production vertex, not the
    // artificial one; the stable photon at the same vertex stays there too.
    const uint32_t muon = findParticleWithPdgId(output, 13);
    const auto muonProductionVertices = output.productionVertices(muon);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), muonProductionVertices.size());
    CPPUNIT_ASSERT(muonProductionVertices.front() != artificial);

    const uint32_t gamma = findParticleWithPdgId(output, 22);
    const auto gammaProductionVertices = output.productionVertices(gamma);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), gammaProductionVertices.size());
    CPPUNIT_ASSERT_EQUAL(muonProductionVertices.front(), gammaProductionVertices.front());

    const uint32_t pion = findParticleWithPdgId(output, 211);
    const auto pionProductionVertices = output.productionVertices(pion);

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), pionProductionVertices.size());
    CPPUNIT_ASSERT_EQUAL(artificial, pionProductionVertices.front());
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testSeedPdgIdZeroKeepsFullGraphForDebugging() {
  try {
    GraphBuilder builder(7, 3);

    // Same topology as the Z -> tau tau test, where the selection would match
    // nothing. The PDG id 0 wildcard must bypass the selection entirely and
    // keep the full graph.
    builder.setGenParticle(0, 23, 2, 100);
    builder.setGenParticle(1, -15, 2, 101);
    builder.setGenParticle(2, 15, 2, 102);
    builder.setGenSimParticle(3, -13, 1, 103, 1003);
    builder.setGenSimParticle(4, 13, 1, 104, 1004);
    builder.setGenParticle(5, -16, 1, 105);
    builder.setGenParticle(6, 16, 1, 106);

    builder.setGenVertex(0, 200);
    builder.setGenSimVertex(1, 201, 2001);
    builder.setGenSimVertex(2, 202, 2002);

    builder.addDecay(0, 0);
    builder.addProduction(0, 1);
    builder.addProduction(0, 2);

    builder.addDecay(1, 1);
    builder.addProduction(1, 3);
    builder.addProduction(1, 5);

    builder.addDecay(2, 2);
    builder.addProduction(2, 4);
    builder.addProduction(2, 6);

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.seedPdgIds = {0};
    config.decayPdgIdGroups = {{13, -13}};

    auto output = runPostProcessing(std::move(graph), config);

    CPPUNIT_ASSERT(output.isConsistent());

    CPPUNIT_ASSERT_EQUAL(uint32_t(7), output.nParticles());
    CPPUNIT_ASSERT_EQUAL(uint32_t(3), output.nVertices());
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 23));
    CPPUNIT_ASSERT(!hasArtificialVertex(output));
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
    //   Z --------^   (unselected, hidden by the seed cut)
    //
    // Unrelated stable final-state e- is kept through the artificial vertex.
    //
    // Then ignoredPdgIds removes pi0, merging H directly to gamma.
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
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 23));
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

    CPPUNIT_ASSERT_EQUAL(std::size_t(1), incoming.size());
    CPPUNIT_ASSERT_EQUAL(int32_t(25), output.particles()[incoming.front()].pdgId);

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

void TestTruthLogicalGraphPostProcessor::testArtificialSourceRolesAndProvenance() {
  try {
    GraphBuilder builder(5, 3);

    // q -> Z -> mu+ mu- ; plus an unrelated stable pi+ (underlying event).
    builder.setGenParticle(0, 1, 2, 100);   // q
    builder.setGenParticle(1, 23, 2, 101);  // Z
    builder.setGenSimParticle(2, -13, 1, 102, 1002);
    builder.setGenSimParticle(3, 13, 1, 103, 1003);
    builder.setGenSimParticle(4, 211, 1, 104, 1004);  // stable spectator

    builder.setGenVertex(0, 200);           // q -> Z (dropped at depth 0)
    builder.setGenSimVertex(1, 201, 2001);  // Z -> mu mu
    builder.setGenVertex(2, 202);           // -> pi+ (dropped)

    builder.addDecay(0, 0);
    builder.addProduction(0, 1);
    builder.addDecay(1, 1);
    builder.addProduction(1, 2);
    builder.addProduction(1, 3);
    builder.addProduction(2, 4);

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.seedPdgIds = {23};
    config.seedParentDepth = 0;
    config.keepStableSpectators = true;

    auto output = runPostProcessing(std::move(graph), config);
    CPPUNIT_ASSERT(output.isConsistent());

    // Z (root with truncated upstream) -> Upstream node; pi+ -> UnderlyingEvent
    // node; both descend from a single Interaction node for the one interaction.
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countArtificialVerticesWithRole(output, truth::VertexRole::Upstream));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countArtificialVerticesWithRole(output, truth::VertexRole::UnderlyingEvent));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countArtificialVerticesWithRole(output, truth::VertexRole::Interaction));

    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 1));  // q dropped at depth 0
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 23));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 211));

    // Provenance: artificial sources carry the genEvent of the activity they summarize.
    for (auto const& vertex : output.vertices()) {
      if (vertex.isArtificial())
        CPPUNIT_ASSERT_EQUAL(int32_t(0), vertex.genEvent);
    }

    const uint32_t z = findParticleWithPdgId(output, 23);
    const auto zProd = output.productionVertices(z);
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), zProd.size());
    CPPUNIT_ASSERT(output.vertices()[zProd.front()].vertexRole() == truth::VertexRole::Upstream);

    // The Upstream and UnderlyingEvent vertices each descend from the single
    // Interaction vertex through one artificial connector particle.
    const uint32_t interaction = findVertexWithRole(output, truth::VertexRole::Interaction);
    const uint32_t upstream = findVertexWithRole(output, truth::VertexRole::Upstream);
    const uint32_t underlyingEvent = findVertexWithRole(output, truth::VertexRole::UnderlyingEvent);

    for (const uint32_t sub : {upstream, underlyingEvent}) {
      const auto incoming = output.incomingParticles(sub);
      CPPUNIT_ASSERT_EQUAL(std::size_t(1), incoming.size());  // the connector particle
      const auto connectorProd = output.productionVertices(incoming.front());
      CPPUNIT_ASSERT_EQUAL(std::size_t(1), connectorProd.size());
      CPPUNIT_ASSERT_EQUAL(interaction, connectorProd.front());
    }
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testKeepStableSpectatorsFalseDropsSpectators() {
  try {
    GraphBuilder builder(5, 3);

    builder.setGenParticle(0, 1, 2, 100);
    builder.setGenParticle(1, 23, 2, 101);
    builder.setGenSimParticle(2, -13, 1, 102, 1002);
    builder.setGenSimParticle(3, 13, 1, 103, 1003);
    builder.setGenSimParticle(4, 211, 1, 104, 1004);  // stable spectator

    builder.setGenVertex(0, 200);
    builder.setGenSimVertex(1, 201, 2001);
    builder.setGenVertex(2, 202);

    builder.addDecay(0, 0);
    builder.addProduction(0, 1);
    builder.addDecay(1, 1);
    builder.addProduction(1, 2);
    builder.addProduction(1, 3);
    builder.addProduction(2, 4);

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.seedPdgIds = {23};
    config.seedParentDepth = 0;
    config.keepStableSpectators = false;

    auto output = runPostProcessing(std::move(graph), config);
    CPPUNIT_ASSERT(output.isConsistent());

    // Spectator pion dropped; no UnderlyingEvent node.
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 211));
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countArtificialVerticesWithRole(output, truth::VertexRole::UnderlyingEvent));

    // Focused subgraph: Z + two muons + one artificial connector, the Z hanging
    // off an Upstream (ISR) node that descends from the Interaction node.
    CPPUNIT_ASSERT_EQUAL(uint32_t(4), output.nParticles());
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 23));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countArtificialVerticesWithRole(output, truth::VertexRole::Upstream));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countArtificialVerticesWithRole(output, truth::VertexRole::Interaction));
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testSeedHadronFlavorSelectsBHadron() {
  try {
    GraphBuilder builder(5, 3);

    // b -> B0 -> D- mu+ ; unrelated stable pi+.
    builder.setGenParticle(0, 5, 2, 100);              // b quark (not a hadron)
    builder.setGenParticle(1, 511, 2, 101);            // B0 (b-hadron)
    builder.setGenSimParticle(2, -411, 2, 102, 1002);  // D-
    builder.setGenSimParticle(3, -13, 1, 103, 1003);   // mu+
    builder.setGenSimParticle(4, 211, 1, 104, 1004);   // stable spectator

    builder.setGenVertex(0, 200);           // b -> B0
    builder.setGenSimVertex(1, 201, 2001);  // B0 -> D- mu+
    builder.setGenVertex(2, 202);           // -> pi+

    builder.addDecay(0, 0);
    builder.addProduction(0, 1);
    builder.addDecay(1, 1);
    builder.addProduction(1, 2);
    builder.addProduction(1, 3);
    builder.addProduction(2, 4);

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.seedHadronFlavors = {5};  // seed all b-hadrons
    config.seedParentDepth = 0;
    config.keepStableSpectators = false;

    auto output = runPostProcessing(std::move(graph), config);
    CPPUNIT_ASSERT(output.isConsistent());

    // The B0 (flavor-5 hadron) is the seed; its decay products are kept.
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 511));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, -411));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, -13));

    // The bare b quark is NOT a hadron, so it is not a seed and is dropped at depth 0.
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 5));
    // Spectator dropped (keepStableSpectators = false).
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 211));
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testJetOriginLowestCommonAncestor() {
  try {
    GraphBuilder builder(5, 2);

    // ttbar-like: top -> W+ b ; b -> pi+ pi- (a b-jet's truth constituents).
    builder.setGenParticle(0, 6, 2, 100);              // top
    builder.setGenParticle(1, 24, 2, 101);             // W+
    builder.setGenParticle(2, 5, 2, 102);              // b
    builder.setGenSimParticle(3, 211, 1, 103, 1003);   // pi+
    builder.setGenSimParticle(4, -211, 1, 104, 1004);  // pi-

    builder.setGenVertex(0, 200);
    builder.setGenSimVertex(1, 201, 2001);

    builder.addDecay(0, 0);
    builder.addProduction(0, 1);
    builder.addProduction(0, 2);
    builder.addDecay(2, 1);
    builder.addProduction(1, 3);
    builder.addProduction(1, 4);

    auto graph = builder.finish();

    // The b-jet constituents come from the b quark (closest common origin).
    auto lcaB = graph.lowestCommonAncestor({graph.particle(3), graph.particle(4)});
    CPPUNIT_ASSERT(lcaB.has_value());
    CPPUNIT_ASSERT_EQUAL(int32_t(5), lcaB->pdgId());

    // Walk up to the originating top.
    auto top = graph.particle(3).firstAncestorWithPdgId(6);
    CPPUNIT_ASSERT(top.has_value());
    CPPUNIT_ASSERT_EQUAL(int32_t(6), top->pdgId());

    // Mixing constituents from the b and W sides yields the top as common origin.
    auto lcaTop = graph.lowestCommonAncestor({graph.particle(3), graph.particle(4), graph.particle(1)});
    CPPUNIT_ASSERT(lcaTop.has_value());
    CPPUNIT_ASSERT_EQUAL(int32_t(6), lcaTop->pdgId());
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testHitlessSimSubgraphsAreDropped() {
  try {
    // pi+ -> {gamma, n}, and n -> {p, nu}. Only the photon leaves a hit.
    //
    //   V0 --> pi+(0) --V1--> gamma(1)   [hit]
    //                    \---> n(2) --V2--> p(3)      [no hit]
    //                                  \--> nu(4)     [GEN-only, no hit]
    //
    // The neutron's whole subgraph (n, p) is hitless, so it must be dropped
    // together with its GEN-only daughter neutrino. The pi+ has no hit of its
    // own but keeps a hit-bearing descendant (the photon), so it survives.
    GraphBuilder builder(5, 3);

    builder.setSimParticle(0, 211, 1000);   // pi+
    builder.setSimParticle(1, 22, 1001);    // gamma (carries a hit)
    builder.setSimParticle(2, 2112, 1002);  // neutron
    builder.setSimParticle(3, 2212, 1003);  // proton
    builder.setGenParticle(4, 12, 1, 104);  // nu_e: GEN-only daughter of the neutron

    builder.setSimVertex(0, 2000);
    builder.setSimVertex(1, 2001);
    builder.setSimVertex(2, 2002);

    builder.addProduction(0, 0);  // source vertex produces the pi+

    builder.addDecay(0, 1);       // pi+ decays at V1
    builder.addProduction(1, 1);  // -> gamma
    builder.addProduction(1, 2);  // -> neutron

    builder.addDecay(2, 2);       // neutron decays at V2
    builder.addProduction(2, 3);  // -> proton
    builder.addProduction(2, 4);  // -> neutrino

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.dropHitlessSimSubgraphs = true;

    // particleDirectHit aligned to the input ids: only the photon (id 1).
    const std::vector<uint8_t> particleDirectHit = {0, 1, 0, 0, 0};

    auto output = runPostProcessing(graph, config, particleDirectHit);

    CPPUNIT_ASSERT(output.isConsistent());

    // pi+ (hitless itself, hit-bearing descendant) and the photon survive.
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 211));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 22));

    // The hitless neutron subgraph, including the GEN-only neutrino, is gone.
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 2112));
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 2212));
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 12));
    CPPUNIT_ASSERT_EQUAL(uint32_t(2), output.nParticles());

    // The kept pi+ still points at the photon through its decay vertex, and the
    // emptied neutron decay vertex has been dropped.
    const uint32_t pion = findParticleWithPdgId(output, 211);
    const uint32_t photon = findParticleWithPdgId(output, 22);
    const auto decayVertices = output.decayVertices(pion);
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), decayVertices.size());
    const auto outgoing = output.outgoingParticles(decayVertices.front());
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), outgoing.size());
    CPPUNIT_ASSERT(std::find(outgoing.begin(), outgoing.end(), photon) != outgoing.end());

    // Without a presence vector the pruning is a no-op: the full graph survives.
    auto untouched = runPostProcessing(graph, config);
    CPPUNIT_ASSERT_EQUAL(uint32_t(5), untouched.nParticles());
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testAttachSelectionSourcesFalseRootsSeedsDirectly() {
  try {
    // q -> Z -> mu+ mu- ; plus an unrelated stable pi+ (underlying event). Same
    // graph as the artificial-source test, but the selection is rooted directly
    // at the seed: no upstream/underlying-event context and no artificial source.
    GraphBuilder builder(5, 3);

    builder.setGenParticle(0, 1, 2, 100);   // q
    builder.setGenParticle(1, 23, 2, 101);  // Z
    builder.setGenSimParticle(2, -13, 1, 102, 1002);
    builder.setGenSimParticle(3, 13, 1, 103, 1003);
    builder.setGenSimParticle(4, 211, 1, 104, 1004);  // stable spectator

    builder.setGenVertex(0, 200);
    builder.setGenSimVertex(1, 201, 2001);
    builder.setGenVertex(2, 202);

    builder.addDecay(0, 0);
    builder.addProduction(0, 1);
    builder.addDecay(1, 1);
    builder.addProduction(1, 2);
    builder.addProduction(1, 3);
    builder.addProduction(2, 4);

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.seedPdgIds = {23};
    config.seedParentDepth = 0;
    config.keepStableSpectators = false;
    config.attachSelectionSources = false;

    auto output = runPostProcessing(std::move(graph), config);
    CPPUNIT_ASSERT(output.isConsistent());

    // No artificial source vertices at all: the Z is a true graph root.
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countArtificialVerticesWithRole(output, truth::VertexRole::Upstream));
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countArtificialVerticesWithRole(output, truth::VertexRole::UnderlyingEvent));
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countArtificialVerticesWithRole(output, truth::VertexRole::Interaction));

    // q dropped (depth 0), pi+ dropped (no spectators); only Z + the two muons remain.
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 1));
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 211));
    CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 23));
    CPPUNIT_ASSERT_EQUAL(uint32_t(2), countParticlesWithPdgId(output, 13) + countParticlesWithPdgId(output, -13));

    // The seed has no production vertex: the subgraph starts directly at the Z.
    const uint32_t z = findParticleWithPdgId(output, 23);
    CPPUNIT_ASSERT(output.productionVertices(z).empty());
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testEventIdKeyingSplitsInteractions() {
  try {
    // Two q -> Z -> mu+ mu- chains tagged with different EncodedEventIds: one
    // signal (eid 0), one pile-up (eid != 0). Each Z is a truncated root, so the
    // per-interaction Interaction vertices are keyed by eventId and the two
    // interactions stay separate - signal + one pile-up interaction.
    GraphBuilder builder(8, 4);

    const uint64_t signalEid = 0;
    const uint64_t pileupEid = 0x2a;  // a distinct pile-up EncodedEventId

    builder.setGenParticle(0, 1, 2, 100);   // q (signal)
    builder.setGenParticle(1, 23, 2, 101);  // Z (signal)
    builder.setGenSimParticle(2, -13, 1, 102, 1002);
    builder.setGenSimParticle(3, 13, 1, 103, 1003);

    builder.setGenParticle(4, 1, 2, 104);   // q (pile-up)
    builder.setGenParticle(5, 23, 2, 105);  // Z (pile-up)
    builder.setGenSimParticle(6, -13, 1, 106, 1006);
    builder.setGenSimParticle(7, 13, 1, 107, 1007);

    // Tag the pile-up chain with its EncodedEventId.
    for (const uint32_t i : {4u, 5u, 6u, 7u})
      builder.graph.particles()[i].eventId = pileupEid;

    builder.setGenVertex(0, 200);           // q -> Z (signal, dropped at depth 0)
    builder.setGenSimVertex(1, 201, 2001);  // Z -> mu mu (signal)
    builder.setGenVertex(2, 202);           // q -> Z (pile-up, dropped at depth 0)
    builder.setGenSimVertex(3, 203, 2003);  // Z -> mu mu (pile-up)

    builder.addDecay(0, 0);
    builder.addProduction(0, 1);
    builder.addDecay(1, 1);
    builder.addProduction(1, 2);
    builder.addProduction(1, 3);

    builder.addDecay(4, 2);
    builder.addProduction(2, 5);
    builder.addDecay(5, 3);
    builder.addProduction(3, 6);
    builder.addProduction(3, 7);

    auto graph = builder.finish();

    auto config = defaultConfig();
    config.seedPdgIds = {23};
    config.seedParentDepth = 0;
    config.keepStableSpectators = false;

    auto output = runPostProcessing(std::move(graph), config);
    CPPUNIT_ASSERT(output.isConsistent());

    // One Interaction (and one Upstream) vertex per interaction: signal + pile-up.
    CPPUNIT_ASSERT_EQUAL(uint32_t(2), countArtificialVerticesWithRole(output, truth::VertexRole::Interaction));
    CPPUNIT_ASSERT_EQUAL(uint32_t(2), countArtificialVerticesWithRole(output, truth::VertexRole::Upstream));
    CPPUNIT_ASSERT_EQUAL(uint32_t(0), countArtificialVerticesWithRole(output, truth::VertexRole::UnderlyingEvent));
    CPPUNIT_ASSERT_EQUAL(uint32_t(2), countParticlesWithPdgId(output, 23));

    // The two Interaction vertices carry the two distinct EncodedEventIds.
    bool sawSignal = false;
    bool sawPileup = false;
    for (auto const& vertex : output.vertices()) {
      if (vertex.vertexRole() != truth::VertexRole::Interaction)
        continue;
      sawSignal = sawSignal || vertex.eventId == signalEid;
      sawPileup = sawPileup || vertex.eventId == pileupEid;
    }
    CPPUNIT_ASSERT(sawSignal);
    CPPUNIT_ASSERT(sawPileup);
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}

void TestTruthLogicalGraphPostProcessor::testSignalOnlyAndBunchCrossingFilterDropPileup() {
  try {
    // Signal Z -> mu+ mu- (eid 0) plus an out-of-time pile-up Z -> mu+ mu- (bx 1).
    // The pile-up filter is orthogonal to the seed selection, so it composes here
    // with seedPdgIds = {23}.
    const uint64_t pileupEid = EncodedEventId(1, 0).rawId();  // out-of-time pile-up (bunchCrossing 1)

    auto buildGraph = [&]() {
      GraphBuilder builder(8, 4);
      builder.setGenParticle(0, 1, 2, 100);   // q (signal)
      builder.setGenParticle(1, 23, 2, 101);  // Z (signal)
      builder.setGenSimParticle(2, -13, 1, 102, 1002);
      builder.setGenSimParticle(3, 13, 1, 103, 1003);
      builder.setGenParticle(4, 1, 2, 104);   // q (pile-up)
      builder.setGenParticle(5, 23, 2, 105);  // Z (pile-up)
      builder.setGenSimParticle(6, -13, 1, 106, 1006);
      builder.setGenSimParticle(7, 13, 1, 107, 1007);
      for (const uint32_t i : {4u, 5u, 6u, 7u})
        builder.graph.particles()[i].eventId = pileupEid;

      builder.setGenVertex(0, 200);
      builder.setGenSimVertex(1, 201, 2001);
      builder.setGenVertex(2, 202);
      builder.setGenSimVertex(3, 203, 2003);
      builder.addDecay(0, 0);
      builder.addProduction(0, 1);
      builder.addDecay(1, 1);
      builder.addProduction(1, 2);
      builder.addProduction(1, 3);
      builder.addDecay(4, 2);
      builder.addProduction(2, 5);
      builder.addDecay(5, 3);
      builder.addProduction(3, 6);
      builder.addProduction(3, 7);
      return builder.finish();
    };

    // (a) No filter: both interactions are kept.
    {
      auto config = defaultConfig();
      config.seedPdgIds = {23};
      auto output = runPostProcessing(buildGraph(), config);
      CPPUNIT_ASSERT(output.isConsistent());
      CPPUNIT_ASSERT_EQUAL(uint32_t(2), countParticlesWithPdgId(output, 23));
    }

    // (b) signalOnly: only the signal Z survives, the pile-up is dropped.
    {
      auto config = defaultConfig();
      config.seedPdgIds = {23};
      config.signalOnly = true;
      auto output = runPostProcessing(buildGraph(), config);
      CPPUNIT_ASSERT(output.isConsistent());
      CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 23));
      CPPUNIT_ASSERT_EQUAL(uint32_t(0), countParticlesWithPdgId(output, 1));  // the pile-up q is gone too
    }

    // (c) keepBunchCrossings = {0}: the out-of-time (bx 1) pile-up is dropped.
    {
      auto config = defaultConfig();
      config.seedPdgIds = {23};
      config.keepBunchCrossings = {0};
      auto output = runPostProcessing(buildGraph(), config);
      CPPUNIT_ASSERT(output.isConsistent());
      CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 23));
    }

    // (d) The pile-up filter is orthogonal to the seed selection and must work
    // with NO seeds: signalOnly on the full graph keeps the signal interaction
    // untouched and drops the pile-up one. (Regression: previously the filter was
    // folded inside the seed selection, which short-circuits without seeds, so the
    // pile-up Z survived.)
    {
      auto config = defaultConfig();
      config.signalOnly = true;
      auto output = runPostProcessing(buildGraph(), config);
      CPPUNIT_ASSERT(output.isConsistent());
      CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 23));   // only the signal Z
      CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 1));    // signal q kept (full graph)
      CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, 13));   // signal mu-
      CPPUNIT_ASSERT_EQUAL(uint32_t(1), countParticlesWithPdgId(output, -13));  // signal mu+
    }
  } catch (cms::Exception const& ex) {
    std::cerr << ex.what() << std::endl;
    CPPUNIT_ASSERT(false);
  }
}
