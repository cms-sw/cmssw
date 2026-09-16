// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "PhysicsTools/TruthInfo/interface/AssignableTarget.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"

namespace {

  // Minimal CSR graph builder, the same one the other tests in this package use.
  struct GraphBuilder {
    explicit GraphBuilder(uint32_t nParticles, uint32_t nVertices) {
      graph.particles().resize(nParticles);
      graph.vertices().resize(nVertices);
      for (uint32_t v = 0; v < nVertices; ++v) {
        graph.vertices()[v].genNode = 200 + v;
      }
    }
    void addDecay(uint32_t particleId, uint32_t vertexId) {
      d2v.emplace_back(particleId, vertexId);
      v2i.emplace_back(vertexId, particleId);
    }
    void addProduction(uint32_t vertexId, uint32_t particleId) {
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
      csr(graph.nParticles(), d2v, graph.particleToDecayVertexOffsets(), graph.particleToDecayVertices());
      csr(graph.nParticles(), p2v, graph.particleToProductionVertexOffsets(), graph.particleToProductionVertices());
      csr(graph.nVertices(), v2o, graph.vertexToOutgoingParticleOffsets(), graph.vertexToOutgoingParticles());
      csr(graph.nVertices(), v2i, graph.vertexToIncomingParticleOffsets(), graph.vertexToIncomingParticles());
      CPPUNIT_ASSERT(graph.isConsistent());
      return graph;
    }
    truth::Graph graph;
    std::vector<std::pair<uint32_t, uint32_t>> d2v, p2v, v2o, v2i;
  };

  // One interaction with everything the rule has to separate.
  //   p0  beam proton, no production vertex, decays at v0 (the hard scatter)
  //   p1  gluon from v0, decays at v1
  //   p2  Z from v0, decays at v2
  //   p3  pi0 from v1, decays at v3
  //   p4  photon from v3, stable
  //   p5  muon from v2, stable
  //   p6  connector, produced at the artificial vertex v4
  //   p7  pi+ from v1, stable, produced at a normal vertex like p3
  constexpr uint32_t kBeam = 0, kGluon = 1, kZ = 2, kPi0 = 3, kPhoton = 4, kMuon = 5, kConnector = 6, kPion = 7;

  truth::Graph buildInteraction() {
    GraphBuilder b(8, 5);
    b.graph.particles()[kBeam].pdgId = 2212;
    b.graph.particles()[kGluon].pdgId = 21;
    b.graph.particles()[kZ].pdgId = 23;
    b.graph.particles()[kPi0].pdgId = 111;
    b.graph.particles()[kPhoton].pdgId = 22;
    b.graph.particles()[kMuon].pdgId = 13;
    b.graph.particles()[kConnector].pdgId = 0;
    b.graph.particles()[kConnector].role = static_cast<uint8_t>(truth::ParticleRole::Connector);
    b.graph.particles()[kPion].pdgId = 211;
    b.graph.vertices()[4].role = static_cast<uint8_t>(truth::VertexRole::InitialState);

    b.addDecay(kBeam, 0);
    b.addProduction(0, kGluon);
    b.addProduction(0, kZ);
    b.addDecay(kGluon, 1);
    b.addProduction(1, kPi0);
    b.addProduction(1, kPion);
    b.addDecay(kZ, 2);
    b.addProduction(2, kMuon);
    b.addDecay(kPi0, 3);
    b.addProduction(3, kPhoton);
    b.addProduction(4, kConnector);
    return b.finish();
  }

}  // namespace

class TestAssignableTarget : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestAssignableTarget);
  CPPUNIT_TEST(testDetectorParticlesAreAssignable);
  CPPUNIT_TEST(testBookkeepingNodesAreNot);
  CPPUNIT_TEST(testEachClauseCanBeTurnedOff);
  CPPUNIT_TEST(testExtraBarredPdgIdsCoverBothSigns);
  CPPUNIT_TEST_SUITE_END();

public:
  void testDetectorParticlesAreAssignable();
  void testBookkeepingNodesAreNot();
  void testEachClauseCanBeTurnedOff();
  void testExtraBarredPdgIdsCoverBothSigns();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestAssignableTarget);

void TestAssignableTarget::testDetectorParticlesAreAssignable() {
  const auto graph = buildInteraction();
  const truth::AssignableTargetConfig config;
  // The merged-pi0 case the adaptive search exists for: the pi0 is reached by crossing a
  // decay vertex and stays a valid answer, as do its photon and the other final state.
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kPi0, config));
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kPhoton, config));
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kMuon, config));
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kPion, config));
}

void TestAssignableTarget::testBookkeepingNodesAreNot() {
  const auto graph = buildInteraction();
  const truth::AssignableTargetConfig config;
  CPPUNIT_ASSERT(!truth::isAssignableTarget(graph, kBeam, config));
  CPPUNIT_ASSERT(!truth::isAssignableTarget(graph, kGluon, config));
  CPPUNIT_ASSERT(!truth::isAssignableTarget(graph, kZ, config));
  CPPUNIT_ASSERT(!truth::isAssignableTarget(graph, kConnector, config));
  // A particle id past the end is not a target rather than an out-of-range read.
  CPPUNIT_ASSERT(!truth::isAssignableTarget(graph, graph.nParticles(), config));
}

void TestAssignableTarget::testEachClauseCanBeTurnedOff() {
  const auto graph = buildInteraction();
  truth::AssignableTargetConfig config;
  config.excludeBeamParticles = false;
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kBeam, config));
  CPPUNIT_ASSERT(!truth::isAssignableTarget(graph, kGluon, config));

  config = truth::AssignableTargetConfig();
  config.excludePartons = false;
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kGluon, config));

  config = truth::AssignableTargetConfig();
  config.excludeElectroweakBosons = false;
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kZ, config));

  config = truth::AssignableTargetConfig();
  config.excludeSynthetic = false;
  // Still barred: its production vertex is the artificial one.
  CPPUNIT_ASSERT(!truth::isAssignableTarget(graph, kConnector, config));
  config.excludeArtificialProduction = false;
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kConnector, config));
}

void TestAssignableTarget::testExtraBarredPdgIdsCoverBothSigns() {
  const auto graph = buildInteraction();
  truth::AssignableTargetConfig config;
  config.extraBarredPdgIds = {211};
  CPPUNIT_ASSERT(!truth::isAssignableTarget(graph, kPion, config));
  CPPUNIT_ASSERT(truth::isAssignableTarget(graph, kPi0, config));

  auto negative = buildInteraction();
  negative.particles()[kPion].pdgId = -211;
  CPPUNIT_ASSERT(!truth::isAssignableTarget(negative, kPion, config));
}
