// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <cstdint>
#include <cstring>

#include "PhysicsTools/TruthInfo/interface/Branch.h"
#include "PhysicsTools/TruthInfo/interface/BranchSelector.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

namespace {

  uint64_t packEventId(int bx, int ev) {
    EncodedEventId id(bx, ev);
    uint64_t out = 0;
    std::memcpy(&out, &id, sizeof(EncodedEventId));
    return out;
  }

  // Three standalone root particles (no edges): each is its own Branch root.
  //   p0: mu-  pt=50, eta=0,  signal
  //   p1: nu   pt=5,  forward, signal
  //   p2: e-   pt=30, eta~1.7, pile-up (bunchCrossing 1)
  truth::Graph buildParticles() {
    truth::Graph g;
    g.particles.resize(3);
    auto set = [&](uint32_t i, int32_t pdg, double px, double py, double pz, double e, uint64_t eid) {
      auto& p = g.particles[i];
      p.genNode = 100 + i;
      p.pdgId = pdg;
      p.status = 1;
      p.genEvent = 0;
      p.eventId = eid;
      p.momentum = math::XYZTLorentzVectorD(px, py, pz, e);
    };
    set(0, 13, 50., 0., 0., 50., packEventId(0, 0));
    set(1, 14, 0., 5., 100., 100., packEventId(0, 0));
    set(2, 11, 30., 0., 80., 85.44, packEventId(1, 0));

    // empty CSR (no edges) consistent with 3 particles, 0 vertices.
    g.particleToDecayVertexOffsets.assign(4, 0);
    g.particleToProductionVertexOffsets.assign(4, 0);
    g.vertexToOutgoingParticleOffsets.assign(1, 0);
    g.vertexToIncomingParticleOffsets.assign(1, 0);
    CPPUNIT_ASSERT(g.isConsistent());
    return g;
  }

}  // namespace

class TestBranchSelector : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestBranchSelector);
  CPPUNIT_TEST(testPtCut);
  CPPUNIT_TEST(testEtaCut);
  CPPUNIT_TEST(testPdgIdAndCharge);
  CPPUNIT_TEST(testSignalAndInTime);
  CPPUNIT_TEST_SUITE_END();

public:
  void testPtCut();
  void testEtaCut();
  void testPdgIdAndCharge();
  void testSignalAndInTime();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestBranchSelector);

void TestBranchSelector::testPtCut() {
  auto g = buildParticles();
  truth::BranchSelector::Config cfg;
  cfg.ptMin = 10.;
  truth::BranchSelector sel(cfg);
  CPPUNIT_ASSERT(sel(truth::Branch(&g, 0)));   // pt 50
  CPPUNIT_ASSERT(!sel(truth::Branch(&g, 1)));  // pt 5
  CPPUNIT_ASSERT(sel(truth::Branch(&g, 2)));   // pt 30
}

void TestBranchSelector::testEtaCut() {
  auto g = buildParticles();
  truth::BranchSelector::Config cfg;
  cfg.etaMin = -1.0;
  cfg.etaMax = 1.0;
  truth::BranchSelector sel(cfg);
  CPPUNIT_ASSERT(sel(truth::Branch(&g, 0)));   // eta 0
  CPPUNIT_ASSERT(!sel(truth::Branch(&g, 1)));  // forward
  CPPUNIT_ASSERT(!sel(truth::Branch(&g, 2)));  // eta ~1.7
}

void TestBranchSelector::testPdgIdAndCharge() {
  auto g = buildParticles();
  truth::BranchSelector::Config muOnly;
  muOnly.pdgIds = {13};
  CPPUNIT_ASSERT(truth::BranchSelector(muOnly)(truth::Branch(&g, 0)));
  CPPUNIT_ASSERT(!truth::BranchSelector(muOnly)(truth::Branch(&g, 2)));  // e- not in list

  truth::BranchSelector::Config chargedCfg;
  chargedCfg.chargedOnly = true;
  truth::BranchSelector charged(chargedCfg);
  CPPUNIT_ASSERT(charged(truth::Branch(&g, 0)));   // mu- charged
  CPPUNIT_ASSERT(!charged(truth::Branch(&g, 1)));  // nu neutral
  CPPUNIT_ASSERT(charged(truth::Branch(&g, 2)));   // e- charged
}

void TestBranchSelector::testSignalAndInTime() {
  auto g = buildParticles();
  truth::BranchSelector::Config signalCfg;
  signalCfg.signalOnly = true;
  truth::BranchSelector signal(signalCfg);
  CPPUNIT_ASSERT(signal(truth::Branch(&g, 0)));   // bx 0, event 0
  CPPUNIT_ASSERT(signal(truth::Branch(&g, 1)));   // bx 0, event 0
  CPPUNIT_ASSERT(!signal(truth::Branch(&g, 2)));  // pile-up (bx 1)

  truth::BranchSelector::Config intimeCfg;
  intimeCfg.intimeOnly = true;
  truth::BranchSelector intime(intimeCfg);
  CPPUNIT_ASSERT(!intime(truth::Branch(&g, 2)));  // bx 1
  CPPUNIT_ASSERT(truth::Branch(&g, 2).isFromPileup());
}
