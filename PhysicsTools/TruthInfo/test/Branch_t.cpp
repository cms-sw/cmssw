// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "PhysicsTools/TruthInfo/interface/Branch.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"

namespace {

  // Minimal graph builder mirroring the one in the postprocessor test.
  struct GraphBuilder {
    explicit GraphBuilder(uint32_t nParticles, uint32_t nVertices) {
      graph.particles.resize(nParticles);
      graph.vertices.resize(nVertices);
    }
    void setParticle(uint32_t id, int32_t pdgId, int16_t status, double e = 1.0) {
      auto& p = graph.particles[id];
      p.genNode = 100 + id;
      p.simNode = -1;
      p.pdgId = pdgId;
      p.status = status;
      p.genEvent = 0;
      p.eventId = 0;
      p.momentum = math::XYZTLorentzVectorD(0., 0., e, e);
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
      csr(graph.nParticles(), d2v, graph.particleToDecayVertexOffsets, graph.particleToDecayVertices);
      csr(graph.nParticles(), p2v, graph.particleToProductionVertexOffsets, graph.particleToProductionVertices);
      csr(graph.nVertices(), v2o, graph.vertexToOutgoingParticleOffsets, graph.vertexToOutgoingParticles);
      csr(graph.nVertices(), v2i, graph.vertexToIncomingParticleOffsets, graph.vertexToIncomingParticles);
      CPPUNIT_ASSERT(graph.isConsistent());
      return graph;
    }
    truth::Graph graph;
    std::vector<std::pair<uint32_t, uint32_t>> d2v, p2v, v2o, v2i;
  };

  // top -> {W+, b}; W+ -> {mu+, nu_mu}; b -> B0; B0 -> {D-, pi+}
  truth::Graph buildTtbarLike() {
    GraphBuilder b(8, 4);
    b.setParticle(0, 6, 2, 100.);    // top
    b.setParticle(1, 24, 2, 80.);    // W+
    b.setParticle(2, 5, 2, 20.);     // b
    b.setParticle(3, 511, 2, 18.);   // B0
    b.setParticle(4, -13, 1, 40.);   // mu+  (leaf)
    b.setParticle(5, 14, 1, 30.);    // nu_mu (leaf, invisible)
    b.setParticle(6, -411, 1, 10.);  // D-  (leaf)
    b.setParticle(7, 211, 1, 5.);    // pi+ (leaf)
    b.addDecay(0, 0);
    b.addProduction(0, 1);
    b.addProduction(0, 2);
    b.addDecay(1, 1);
    b.addProduction(1, 4);
    b.addProduction(1, 5);
    b.addDecay(2, 2);
    b.addProduction(2, 3);
    b.addDecay(3, 3);
    b.addProduction(3, 6);
    b.addProduction(3, 7);
    return b.finish();
  }

  std::size_t countPdg(std::vector<truth::Particle> const& ps, int32_t pdg) {
    return std::count_if(ps.begin(), ps.end(), [pdg](auto const& p) { return p.pdgId() == pdg; });
  }

}  // namespace

class TestBranch : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestBranch);
  CPPUNIT_TEST(testClosures);
  CPPUNIT_TEST(testKinematics);
  CPPUNIT_TEST(testTaggingAndProvenance);
  CPPUNIT_TEST(testRelations);
  CPPUNIT_TEST_SUITE_END();

public:
  void testClosures();
  void testKinematics();
  void testTaggingAndProvenance();
  void testRelations();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestBranch);

void TestBranch::testClosures() {
  auto g = buildTtbarLike();

  // Subtree from the top: all 8 particles.
  CPPUNIT_ASSERT_EQUAL(std::size_t(8), truth::Branch(&g, 0).members().size());

  // StableLeaves: top (root) + 4 leaves.
  auto leaves = truth::Branch(&g, 0, truth::ClosureSpec::stableLeaves()).members();
  CPPUNIT_ASSERT_EQUAL(std::size_t(5), leaves.size());

  // DepthN(1): top + W + b.
  CPPUNIT_ASSERT_EQUAL(std::size_t(3), truth::Branch(&g, 0, truth::ClosureSpec::depth(1)).members().size());

  // UntilPdgId stopping at the B0: top, W, b, B0, mu+, nu_mu (D-/pi+ excluded).
  auto untilB = truth::Branch(&g, 0, truth::ClosureSpec::untilPdgId({511})).members();
  CPPUNIT_ASSERT_EQUAL(std::size_t(6), untilB.size());
  CPPUNIT_ASSERT_EQUAL(std::size_t(1), countPdg(untilB, 511));
  CPPUNIT_ASSERT_EQUAL(std::size_t(0), countPdg(untilB, -411));

  // Predicate stopping at any b-hadron: same effect as untilPdgId({511}) here.
  auto untilHF = truth::Branch(&g, 0, truth::ClosureSpec::predicate([](truth::Particle p) {
                   const int id = std::abs(p.pdgId());
                   return id > 100 && ((id / 100) % 10 == 5 || (id / 1000) % 10 == 5);
                 })).members();
  CPPUNIT_ASSERT_EQUAL(std::size_t(0), countPdg(untilHF, -411));
}

void TestBranch::testKinematics() {
  auto g = buildTtbarLike();
  truth::Branch top(&g, 0);

  // p4 = sum of stable leaves (mu+ 40, nu 30, D- 10, pi+ 5) = 85 in E.
  CPPUNIT_ASSERT_DOUBLES_EQUAL(85.0, top.energy(), 1e-6);
  // visible excludes the neutrino (30).
  CPPUNIT_ASSERT_DOUBLES_EQUAL(55.0, top.visibleEnergy(), 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(30.0, top.invisibleEnergy(), 1e-6);
}

void TestBranch::testTaggingAndProvenance() {
  auto g = buildTtbarLike();

  truth::Branch bBranch(&g, 2);  // rooted at the b quark
  CPPUNIT_ASSERT_EQUAL(int32_t(5), bBranch.rootPdgId());
  // the b-branch originates from the top.
  auto origin = bBranch.originWithPdgId(6);
  CPPUNIT_ASSERT(origin.has_value());
  CPPUNIT_ASSERT_EQUAL(int32_t(6), origin->pdgId());
  // heavy-flavor content: contains a b-hadron (B0) and a c-hadron (D-).
  CPPUNIT_ASSERT(bBranch.hasHeavyFlavor(5));
  CPPUNIT_ASSERT(bBranch.hasHeavyFlavor(4));

  // a leptonic W branch has no heavy flavor.
  truth::Branch wBranch(&g, 1);
  CPPUNIT_ASSERT(!wBranch.hasHeavyFlavor(5));

  // provenance: built with eventId 0 -> in-time, not pile-up.
  CPPUNIT_ASSERT(truth::Branch(&g, 0).isInTime());
  CPPUNIT_ASSERT(!truth::Branch(&g, 0).isFromPileup());
}

void TestBranch::testRelations() {
  auto g = buildTtbarLike();
  truth::Branch wBranch(&g, 1);
  truth::Branch bBranch(&g, 2);

  // W and b branches share the top as common ancestor.
  auto common = wBranch.commonAncestor(bBranch);
  CPPUNIT_ASSERT(common.has_value());
  CPPUNIT_ASSERT_EQUAL(int32_t(6), common->pdgId());

  // merged branch has both roots and the union of subtrees.
  auto merged = wBranch.merged(bBranch);
  CPPUNIT_ASSERT_EQUAL(std::size_t(2), merged.rootIds().size());
  // W subtree (W,mu,nu) + b subtree (b,B0,D-,pi+) = 7 members.
  CPPUNIT_ASSERT_EQUAL(std::size_t(7), merged.members().size());
}
