// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "PhysicsTools/TruthInfo/interface/Interactions.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"

namespace {

  uint64_t packed(int bunchCrossing, int eventIndex) { return EncodedEventId(bunchCrossing, eventIndex).rawId(); }

  // One particle per interaction, produced at one vertex, so every interaction resolves
  // whether or not the vertices carry the Interaction role.
  struct GraphBuilder {
    void addInteraction(uint64_t eventId,
                        truth::VertexRole role,
                        bool hasSim,
                        math::XYZTLorentzVectorD position = math::XYZTLorentzVectorD(1., 2., 3., 0.)) {
      const uint32_t vertexId = static_cast<uint32_t>(graph.vertices().size());
      truth::VertexData v;
      v.genNode = 100 + static_cast<int32_t>(vertexId);
      v.simNode = hasSim ? static_cast<int32_t>(vertexId) : -1;
      v.eventId = eventId;
      v.role = static_cast<uint8_t>(role);
      v.position = position;
      graph.vertices().push_back(v);

      const uint32_t particleId = static_cast<uint32_t>(graph.particles().size());
      truth::ParticleData p;
      p.genNode = 200 + static_cast<int32_t>(particleId);
      p.simNode = -1;
      p.pdgId = 211;
      p.status = 1;
      p.eventId = eventId;
      graph.particles().push_back(p);

      v2o.emplace_back(vertexId, particleId);
      p2v.emplace_back(particleId, vertexId);
    }

    static void csr(uint32_t n,
                    std::vector<std::pair<uint32_t, uint32_t>>& pairs,
                    std::vector<uint32_t>& off,
                    std::vector<uint32_t>& flat) {
      std::sort(pairs.begin(), pairs.end());
      pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
      off.assign(n + 1, 0);
      for (auto const& pr : pairs)
        ++off[pr.first + 1];
      for (uint32_t i = 1; i <= n; ++i)
        off[i] += off[i - 1];
      flat.assign(pairs.size(), 0);
      auto cur = off;
      for (auto const& pr : pairs)
        flat[cur[pr.first]++] = pr.second;
    }

    truth::Graph finish() {
      std::vector<std::pair<uint32_t, uint32_t>> none;
      csr(graph.nParticles(), none, graph.particleToDecayVertexOffsets(), graph.particleToDecayVertices());
      csr(graph.nParticles(), p2v, graph.particleToProductionVertexOffsets(), graph.particleToProductionVertices());
      csr(graph.nVertices(), v2o, graph.vertexToOutgoingParticleOffsets(), graph.vertexToOutgoingParticles());
      none.clear();
      csr(graph.nVertices(), none, graph.vertexToIncomingParticleOffsets(), graph.vertexToIncomingParticles());
      CPPUNIT_ASSERT(graph.isConsistent());
      return graph;
    }

    truth::Graph graph;
    std::vector<std::pair<uint32_t, uint32_t>> p2v, v2o;
  };

}  // namespace

class TestInteractions : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestInteractions);
  CPPUNIT_TEST(testSignalComesFirst);
  CPPUNIT_TEST(testElectionWithoutInteractionNodes);
  CPPUNIT_TEST(testPlaceholderIsReported);
  CPPUNIT_TEST(testNoSignal);
  CPPUNIT_TEST_SUITE_END();

public:
  void testSignalComesFirst();
  void testElectionWithoutInteractionNodes();
  void testPlaceholderIsReported();
  void testNoSignal();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestInteractions);

void TestInteractions::testSignalComesFirst() {
  // Built out of order, and with the signal's vertex last, so neither the insertion
  // order nor the vertex id can be what puts the signal first.
  GraphBuilder b;
  b.addInteraction(packed(0, 2), truth::VertexRole::Interaction, true);
  b.addInteraction(packed(1, 0), truth::VertexRole::Interaction, true);
  b.addInteraction(packed(-1, 3), truth::VertexRole::Interaction, true);
  b.addInteraction(packed(0, 1), truth::VertexRole::Interaction, true);
  b.addInteraction(packed(0, 0), truth::VertexRole::Interaction, true);
  auto graph = b.finish();

  const auto all = truth::interactions(graph);
  CPPUNIT_ASSERT_EQUAL(std::size_t(5), all.size());

  CPPUNIT_ASSERT(all.front().isSignal());
  CPPUNIT_ASSERT_EQUAL(uint32_t(4), all.front().vertexId);
  for (std::size_t i = 1; i < all.size(); ++i)
    CPPUNIT_ASSERT(!all[i].isSignal());

  // Pile-up by bunch crossing, then by index inside the crossing.
  const std::vector<std::pair<int, int>> expected{{0, 0}, {-1, 3}, {0, 1}, {0, 2}, {1, 0}};
  for (std::size_t i = 0; i < all.size(); ++i) {
    CPPUNIT_ASSERT_EQUAL(expected[i].first, all[i].bunchCrossing());
    CPPUNIT_ASSERT_EQUAL(expected[i].second, all[i].eventIndex());
    CPPUNIT_ASSERT(!all[i].isPlaceholder);
  }

  const auto signal = truth::signalInteraction(graph);
  CPPUNIT_ASSERT(signal.has_value());
  CPPUNIT_ASSERT_EQUAL(uint32_t(4), signal->vertexId);
}

void TestInteractions::testElectionWithoutInteractionNodes() {
  // No preset ran, so every vertex is Normal and the interaction resolves to the
  // production vertex of its own particles.
  GraphBuilder b;
  b.addInteraction(packed(0, 0), truth::VertexRole::Normal, true);
  b.addInteraction(packed(0, 1), truth::VertexRole::Normal, true);
  auto graph = b.finish();

  const auto all = truth::interactions(graph);
  CPPUNIT_ASSERT_EQUAL(std::size_t(2), all.size());
  CPPUNIT_ASSERT(all[0].isSignal());
  CPPUNIT_ASSERT_EQUAL(uint32_t(0), all[0].vertexId);
  CPPUNIT_ASSERT_EQUAL(uint32_t(1), all[1].vertexId);
  CPPUNIT_ASSERT(!all[0].isPlaceholder);
  CPPUNIT_ASSERT(!all[1].isPlaceholder);
}

void TestInteractions::testPlaceholderIsReported() {
  // A vertex that did not merge with a SimVertex and sits at the origin at time zero
  // has no position to offer, so the interaction resolves but is flagged.
  GraphBuilder b;
  b.addInteraction(packed(0, 0), truth::VertexRole::Normal, true);
  b.addInteraction(packed(0, 1), truth::VertexRole::Normal, false, math::XYZTLorentzVectorD());
  // A GEN-only vertex with a nonzero time is a real vertex, not a placeholder.
  b.addInteraction(packed(0, 2), truth::VertexRole::Normal, false, math::XYZTLorentzVectorD(0., 0., 0., 5.));
  auto graph = b.finish();

  const auto all = truth::interactions(graph);
  CPPUNIT_ASSERT_EQUAL(std::size_t(3), all.size());
  CPPUNIT_ASSERT(!all[0].isPlaceholder);
  CPPUNIT_ASSERT(all[1].isPlaceholder);
  CPPUNIT_ASSERT(!all[2].isPlaceholder);
}

void TestInteractions::testNoSignal() {
  GraphBuilder b;
  b.addInteraction(packed(0, 1), truth::VertexRole::Interaction, true);
  b.addInteraction(packed(2, 4), truth::VertexRole::Interaction, true);
  auto graph = b.finish();

  CPPUNIT_ASSERT_EQUAL(std::size_t(2), truth::interactions(graph).size());
  CPPUNIT_ASSERT(!truth::signalInteraction(graph).has_value());

  truth::Graph empty;
  CPPUNIT_ASSERT(truth::interactions(empty).empty());
  CPPUNIT_ASSERT(!truth::signalInteraction(empty).has_value());
}
