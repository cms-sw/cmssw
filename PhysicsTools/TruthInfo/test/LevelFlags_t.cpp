// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "FWCore/Utilities/interface/Exception.h"
#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"

namespace {

  // Minimal CSR graph builder, the same one the other tests in this package use.
  struct GraphBuilder {
    explicit GraphBuilder(uint32_t nParticles, uint32_t nVertices) {
      graph.particles().resize(nParticles);
      graph.vertices().resize(nVertices);
      // Vertices are GEN by default, as the decay vertices of GEN particles are in a
      // production graph; a test exercising SIM continuation overrides genNode to -1.
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

  // A tau decaying to a pion and a neutrino, the pion reaching the calorimeter.
  //   p0  tau, isHardProcess, decays at v0
  //   p1  pi+, status 1, records a calorimeter boundary crossing
  //   p2  nu,  status 1, no crossing
  truth::Graph buildDecay() {
    GraphBuilder b(3, 1);

    auto& tau = b.graph.particles()[0];
    tau.genNode = 100;
    tau.pdgId = 15;
    tau.status = 2;
    tau.statusFlags = truth::detail::kIsHardProcess;
    tau.momentum = math::XYZTLorentzVectorD(50., 0., 0., 60.);

    auto& pion = b.graph.particles()[1];
    pion.genNode = 101;
    pion.simNode = 201;
    pion.pdgId = 211;
    pion.status = 1;
    pion.momentum = math::XYZTLorentzVectorD(30., 0., 0., 35.);
    pion.checkpoints.push_back(truth::Checkpoint{});

    auto& nu = b.graph.particles()[2];
    nu.genNode = 102;
    nu.pdgId = 16;
    nu.status = 1;
    nu.momentum = math::XYZTLorentzVectorD(5., 0., 0., 5.);

    b.addDecay(0, 0);
    b.addProduction(0, 1);
    b.addProduction(0, 2);
    return b.finish();
  }

  // A semileptonic top decay chain, the shape the parton jet level has to get right.
  //   p0  g,    incoming beam parton at beam rapidity, decays at v2
  //   p1  t,    decays at v0
  //   p2  b,    outgoing leg, no children
  //   p3  W+,   decays at v1
  //   p4  u,    outgoing leg, no children
  //   p5  dbar, outgoing leg, no children
  // Every one of them carries isHardProcess, which is what makes this the discriminating
  // fixture: the level has to separate them on ancestry and species, not on the flag.
  truth::Graph buildTopDecay() {
    GraphBuilder b(6, 3);

    auto set = [&](uint32_t id, int32_t pdgId, int16_t status, math::XYZTLorentzVectorD p4) {
      auto& particle = b.graph.particles()[id];
      particle.genNode = 100 + static_cast<int32_t>(id);
      particle.pdgId = pdgId;
      particle.status = status;
      particle.statusFlags = truth::detail::kIsHardProcess;
      particle.momentum = p4;
    };

    set(0, 21, 21, math::XYZTLorentzVectorD(0., 0., 857., 857.));
    set(1, 6, 22, math::XYZTLorentzVectorD(60., 20., 30., 190.));
    set(2, 5, 23, math::XYZTLorentzVectorD(10., 1., 18., 21.));
    set(3, 24, 22, math::XYZTLorentzVectorD(50., 19., 12., 100.));
    set(4, 2, 23, math::XYZTLorentzVectorD(-34., -6., -150., 154.));
    set(5, -1, 23, math::XYZTLorentzVectorD(-30., 70., -139., 159.));

    b.addDecay(0, 2);
    b.addProduction(2, 1);
    b.addDecay(1, 0);
    b.addProduction(0, 2);
    b.addProduction(0, 3);
    b.addDecay(3, 1);
    b.addProduction(1, 4);
    b.addProduction(1, 5);
    return b.finish();
  }

  // A b quark fragmenting to a B* that radiates to a B, which decays to a D, which decays
  // to a kaon and a pion. Both the generator-copy nesting (B* above B) and the
  // cross-flavour nesting (B above D) are present, since those are the two things the
  // heavy-flavour levels have to get right.
  //   p0 b -> v0 -> p1 B*+ -> v1 -> p2 B+ -> v2 -> p3 D0bar -> v3 -> p4 K+, p5 pi-
  truth::Graph buildHeavyFlavour() {
    GraphBuilder b(6, 4);

    auto set = [&](uint32_t id, int32_t pdgId, int16_t status) {
      auto& particle = b.graph.particles()[id];
      particle.genNode = 300 + static_cast<int32_t>(id);
      particle.pdgId = pdgId;
      particle.status = status;
      particle.momentum = math::XYZTLorentzVectorD(10., 2., 5., 12.);
    };

    set(0, 5, 23);
    b.graph.particles()[0].statusFlags = truth::detail::kIsHardProcess;
    set(1, 523, 2);
    set(2, 521, 2);
    set(3, -421, 2);
    set(4, 321, 1);
    set(5, -211, 1);

    b.addDecay(0, 0);
    b.addProduction(0, 1);
    b.addDecay(1, 1);
    b.addProduction(1, 2);
    b.addDecay(2, 2);
    b.addProduction(2, 3);
    b.addDecay(3, 3);
    b.addProduction(3, 4);
    b.addProduction(3, 5);
    return b.finish();
  }

  // A radiating tau chain ending hadronically, a leptonic tau, and a pi0 with its two
  // photons, all GEN, no Signal flag anywhere. Ids:
  //   0 tau (radiates)  1 tau (last copy, hadronic)  2 photon (radiated)
  //   3 pi+  4 nu_tau   5 tau (leptonic)  6 mu  7 nu_tau  8 nu_mu
  //   9 pi0  10 photon  11 photon
  truth::Graph buildTausAndPi0() {
    GraphBuilder b(12, 6);
    auto set = [&](uint32_t id, int32_t pdgId, int16_t status) {
      auto& particle = b.graph.particles()[id];
      particle.genNode = 400 + static_cast<int32_t>(id);
      particle.pdgId = pdgId;
      particle.status = status;
      particle.momentum = math::XYZTLorentzVectorD(10., 1., 3., 11.);
    };
    set(0, 15, 2);
    set(1, 15, 2);
    set(2, 22, 1);
    set(3, 211, 1);
    set(4, 16, 1);
    set(5, -15, 2);
    set(6, -13, 1);
    set(7, 16, 1);
    set(8, -14, 1);
    set(9, 111, 2);
    set(10, 22, 1);
    set(11, 22, 1);

    b.addDecay(0, 0);
    b.addProduction(0, 1);
    b.addProduction(0, 2);
    b.addDecay(1, 1);
    b.addProduction(1, 3);
    b.addProduction(1, 4);
    b.addDecay(5, 2);
    b.addProduction(2, 6);
    b.addProduction(2, 7);
    b.addProduction(2, 8);
    b.addDecay(9, 3);
    b.addProduction(3, 10);
    b.addProduction(3, 11);
    return b.finish();
  }

}  // namespace

class LevelFlags_t : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(LevelFlags_t);
  CPPUNIT_TEST(testFitsInThePaddingHole);
  CPPUNIT_TEST(testFlagsMatchTheAntichain);
  CPPUNIT_TEST(testIdempotent);
  CPPUNIT_TEST(testSignalSurvivesFillLevelFlags);
  CPPUNIT_TEST(testSyntheticSignalNodeIsMarked);
  CPPUNIT_TEST(testConnectorIsNotAStandIn);
  CPPUNIT_TEST(testReconstructableFromSignalDropsNeutrinos);
  CPPUNIT_TEST(testPi0IsLabelledNotItsPhotons);
  CPPUNIT_TEST(testGunSeedThatIsItselfReconstructableIsItsOwnLeg);
  CPPUNIT_TEST(testSimContinuationIsNotADecay);
  CPPUNIT_TEST(testReconvergentHistoryStaysAnAntichain);
  CPPUNIT_TEST(testStableLegsFromUpstreamAndUnderlyingEvent);
  CPPUNIT_TEST(testDiquarksAreNotHeavyFlavourHadrons);
  CPPUNIT_TEST(testThreeProngThroughAnIntermediateResonance);
  CPPUNIT_TEST(testPartonJetsKeepTheQuarksNotTheTopOrTheBeam);
  CPPUNIT_TEST(testPartonJetsExcludeLeptons);
  CPPUNIT_TEST(testHeavyFlavourKeepsTheWeaklyDecayingHadron);
  CPPUNIT_TEST(testVisibleTauIsTheLastHadronicCopy);
  CPPUNIT_TEST(testReconstructableFinalStateNeedsNoSignal);
  CPPUNIT_TEST(testBeautyAndCharmAreSeparateLevels);
  CPPUNIT_TEST(testEmptyGraphStampsNothing);
  CPPUNIT_TEST(testAcyclicGraphReportsNoCycle);
  CPPUNIT_TEST(testCycleIsReported);
  CPPUNIT_TEST(testCycleDoesNotStopStamping);
  CPPUNIT_TEST(testLevelTableIsTheSingleSource);
  CPPUNIT_TEST_SUITE_END();

public:
  // REQUIRED: the flags word occupies the alignment hole between genEvent and momentum,
  // so carrying it costs no memory. It is not free ON DISK, where ROOT streams members
  // and not padding. A change that grows ParticleData past 96 bytes has moved it out of
  // the hole and needs to be justified, not absorbed silently.
  void testFitsInThePaddingHole() {
    CPPUNIT_ASSERT_EQUAL(std::size_t{96}, sizeof(truth::ParticleData));
    // sizeof alone does not pin the members down: six bytes of tail padding follow
    // role, so widening role or dropping a member keeps sizeof at 96. Pin the two
    // offsets that carry the claim: levelFlags fills the hole after genEvent, and role
    // sits in the tail after backscattered.
    truth::ParticleData d;
    auto const* base = reinterpret_cast<char const*>(&d);
    CPPUNIT_ASSERT_EQUAL(std::ptrdiff_t{28}, reinterpret_cast<char const*>(&d.levelFlags) - base);
    CPPUNIT_ASSERT_EQUAL(std::ptrdiff_t{89}, reinterpret_cast<char const*>(&d.role) - base);
  }

  // REQUIRED: a stored flag says exactly what levelAntichain() would say. This is the
  // defence against a graph written before a level definition changed, which is
  // indistinguishable from a fresh one by inspection.
  void testFlagsMatchTheAntichain() {
    truth::Graph g = buildDecay();
    truth::fillLevelFlags(g);

    for (const truth::Level level : truth::kAllLevels) {
      std::vector<bool> expected(g.nParticles(), false);
      for (const uint32_t id : truth::levelAntichain(g, level)) {
        expected[id] = true;
      }
      const truth::LevelFlag flag = truth::levelFlagOf(level);
      for (uint32_t id = 0; id < g.nParticles(); ++id) {
        CPPUNIT_ASSERT_EQUAL(static_cast<bool>(expected[id]), g.particles()[id].isAtLevel(flag));
      }
    }

    // The physics the sample encodes, so a passing test means the right thing and not
    // merely a self-consistent one: the tau is the hard process, the pion and the
    // neutrino are the stable decay products, only the pion reaches the calorimeter.
    CPPUNIT_ASSERT(g.particles()[0].isAtLevel(truth::LevelFlag::HardProcess));
    CPPUNIT_ASSERT(!g.particles()[1].isAtLevel(truth::LevelFlag::HardProcess));
    CPPUNIT_ASSERT(g.particles()[1].isAtLevel(truth::LevelFlag::StableDecayProducts));
    CPPUNIT_ASSERT(g.particles()[2].isAtLevel(truth::LevelFlag::StableDecayProducts));
    CPPUNIT_ASSERT(g.particles()[1].isAtLevel(truth::LevelFlag::CaloBoundary));
    CPPUNIT_ASSERT(!g.particles()[2].isAtLevel(truth::LevelFlag::CaloBoundary));
  }

  // REQUIRED: filling twice leaves the same answer as filling once, so a graph that
  // passes through the stamp again cannot accumulate membership it no longer has.
  void testIdempotent() {
    truth::Graph g = buildDecay();
    truth::fillLevelFlags(g);
    std::vector<uint32_t> once;
    for (auto const& p : g.particles()) {
      once.push_back(p.levelFlags);
    }
    truth::fillLevelFlags(g);
    for (uint32_t id = 0; id < g.nParticles(); ++id) {
      CPPUNIT_ASSERT_EQUAL(once[id], g.particles()[id].levelFlags);
    }
  }

  // REQUIRED: Signal is set by the selection post-processing, not by fillLevelFlags, so
  // fillLevelFlags must clear only its own four bits. Clearing everything would erase the
  // resonance and nothing downstream would notice.
  void testSignalSurvivesFillLevelFlags() {
    truth::Graph g = buildDecay();
    g.particles()[0].setLevel(truth::LevelFlag::Signal);
    truth::fillLevelFlags(g);
    CPPUNIT_ASSERT(g.particles()[0].isAtLevel(truth::LevelFlag::Signal));
    // and the levels it does own are still correct alongside it
    CPPUNIT_ASSERT(g.particles()[0].isAtLevel(truth::LevelFlag::HardProcess));
    CPPUNIT_ASSERT(!g.particles()[1].isAtLevel(truth::LevelFlag::Signal));
  }

  // REQUIRED: the synthetic stand-in is distinguishable from every real particle, since
  // nothing may read its momentum as a generator quantity. No GEN, no SIM, status 0.
  // This path fires on no sample in the current set, so this test is the only thing
  // exercising it.
  void testSyntheticSignalNodeIsMarked() {
    truth::Graph g = buildDecay();
    const uint32_t before = g.nParticles();

    truth::ParticleData synthetic;
    synthetic.role = static_cast<uint8_t>(truth::ParticleRole::SignalStandIn);
    synthetic.genNode = -1;
    synthetic.simNode = -1;
    synthetic.status = 0;
    synthetic.setLevel(truth::LevelFlag::Signal);
    g.particles().push_back(synthetic);
    g.particleToDecayVertexOffsets().push_back(g.particleToDecayVertexOffsets().back());
    g.particleToProductionVertexOffsets().push_back(g.particleToProductionVertexOffsets().back());

    CPPUNIT_ASSERT_EQUAL(before + 1, g.nParticles());
    CPPUNIT_ASSERT(g.isConsistent());
    auto const& s = g.particles()[before];
    CPPUNIT_ASSERT(s.isAtLevel(truth::LevelFlag::Signal));
    CPPUNIT_ASSERT(s.isSynthetic());
    CPPUNIT_ASSERT(truth::ParticleRole::SignalStandIn == s.particleRole());
    // Every real particle in the fixture is distinguishable from it.
    for (uint32_t id = 0; id < before; ++id) {
      CPPUNIT_ASSERT(!g.particles()[id].isSynthetic());
    }
    // fillLevelFlags must leave a standalone synthetic node alone apart from its own bits.
    truth::fillLevelFlags(g);
    CPPUNIT_ASSERT(g.particles()[before].isAtLevel(truth::LevelFlag::Signal));
  }

  // REQUIRED: a connector and a signal stand-in must be distinguishable. Both are
  // synthetic and both carry genNode = simNode = -1, pdgId 0 and status 0, so any test
  // that infers the kind from those empty fields cannot tell them apart. That is exactly
  // the bug this case exists to prevent.
  void testConnectorIsNotAStandIn() {
    truth::ParticleData connector;
    connector.genNode = -1;
    connector.simNode = -1;
    connector.pdgId = 0;
    connector.status = 0;
    connector.role = static_cast<uint8_t>(truth::ParticleRole::Connector);

    truth::ParticleData standIn;
    standIn.genNode = -1;
    standIn.simNode = -1;
    standIn.pdgId = 0;
    standIn.status = 0;
    standIn.role = static_cast<uint8_t>(truth::ParticleRole::SignalStandIn);

    // Indistinguishable on the fields alone, which is the point.
    CPPUNIT_ASSERT_EQUAL(connector.genNode, standIn.genNode);
    CPPUNIT_ASSERT_EQUAL(connector.simNode, standIn.simNode);
    CPPUNIT_ASSERT_EQUAL(connector.status, standIn.status);
    // Distinguishable on the role, which is why the role exists.
    CPPUNIT_ASSERT(connector.isSynthetic());
    CPPUNIT_ASSERT(standIn.isSynthetic());
    CPPUNIT_ASSERT(connector.role != standIn.role);

    truth::ParticleData real;
    real.genNode = 7;
    CPPUNIT_ASSERT(!real.isSynthetic());
  }

  // REQUIRED: from a signal root, the first GEN-stable descendants, with neutrinos
  // dropped because they cannot be reconstructed. The fixture tau decays to a pion and a
  // neutrino, so the visible final state is the pion alone. Empty when nothing is Signal.
  void testReconstructableFromSignalDropsNeutrinos() {
    truth::Graph g = buildDecay();

    // No Signal flag anywhere: the level is empty, not the whole event.
    CPPUNIT_ASSERT(truth::reconstructableFromSignal(g).empty());

    g.particles()[0].setLevel(truth::LevelFlag::Signal);
    truth::fillLevelFlags(g);

    const auto legs = truth::reconstructableFromSignal(g);
    CPPUNIT_ASSERT_EQUAL(std::size_t{1}, legs.size());
    CPPUNIT_ASSERT_EQUAL(uint32_t{1}, legs[0]);  // the pion
    CPPUNIT_ASSERT(g.particles()[1].isAtLevel(truth::LevelFlag::ReconstructableFromSignal));
    // the neutrino is stable but invisible, so it is not a leg
    CPPUNIT_ASSERT(!g.particles()[2].isAtLevel(truth::LevelFlag::ReconstructableFromSignal));
    // the tau decayed, so it is not its own leg
    CPPUNIT_ASSERT(!g.particles()[0].isAtLevel(truth::LevelFlag::ReconstructableFromSignal));
  }

  // REQUIRED: a pi0 decays to two photons at once, but it is the pi0 the analysis
  // reconstructs, so the pi0 is labelled and its photons are not.
  //   p0 tau (signal) -> v0 -> p1 pi0 -> v1 -> p2 gamma, p3 gamma
  void testPi0IsLabelledNotItsPhotons() {
    GraphBuilder b(4, 2);
    auto set = [&](uint32_t i, int32_t pdg, int16_t st) {
      auto& d = b.graph.particles()[i];
      d.genNode = 100 + i;
      d.pdgId = pdg;
      d.status = st;
      d.momentum = math::XYZTLorentzVectorD(10., 0., 0., 10.);
    };
    set(0, 15, 2);   // tau, decays
    set(1, 111, 2);  // pi0, decays, but reconstructable as an object
    set(2, 22, 1);   // gamma
    set(3, 22, 1);   // gamma
    b.addDecay(0, 0);
    b.addProduction(0, 1);
    b.addDecay(1, 1);
    b.addProduction(1, 2);
    b.addProduction(1, 3);
    truth::Graph g = b.finish();
    g.reconstructablePdgIds() = {111};
    g.particles()[0].setLevel(truth::LevelFlag::Signal);
    truth::fillLevelFlags(g);

    const auto legs = truth::reconstructableFromSignal(g);
    CPPUNIT_ASSERT_EQUAL(std::size_t{1}, legs.size());
    CPPUNIT_ASSERT_EQUAL(uint32_t{1}, legs[0]);
    CPPUNIT_ASSERT(g.particles()[1].isAtLevel(truth::LevelFlag::ReconstructableFromSignal));
    CPPUNIT_ASSERT(!g.particles()[2].isAtLevel(truth::LevelFlag::ReconstructableFromSignal));
    CPPUNIT_ASSERT(!g.particles()[3].isAtLevel(truth::LevelFlag::ReconstructableFromSignal));

    // With the pi0 NOT declared reconstructable the walk goes through it to the photons,
    // which is the control showing the configuration is what decides, not the code.
    g.reconstructablePdgIds().clear();
    const auto through = truth::reconstructableFromSignal(g);
    CPPUNIT_ASSERT_EQUAL(std::size_t{2}, through.size());
  }

  // The gun shape: the Signal root IS the reconstructable species (a pi0 gun seeds
  // 111). The walk terminates on the root itself, so the gun particle is one object
  // and its photons are never labelled, the same rule a pi0 inside a tau decay gets.
  void testGunSeedThatIsItselfReconstructableIsItsOwnLeg() {
    GraphBuilder b(3, 1);
    auto set = [&](uint32_t i, int32_t pdg, int16_t st) {
      auto& d = b.graph.particles()[i];
      d.genNode = 100 + i;
      d.pdgId = pdg;
      d.status = st;
      d.momentum = math::XYZTLorentzVectorD(10., 0., 0., 10.);
    };
    set(0, 111, 2);  // the gun pi0, decays
    set(1, 22, 1);   // gamma
    set(2, 22, 1);   // gamma
    b.addDecay(0, 0);
    b.addProduction(0, 1);
    b.addProduction(0, 2);
    truth::Graph g = b.finish();
    g.reconstructablePdgIds() = {111};
    g.particles()[0].setLevel(truth::LevelFlag::Signal);
    truth::fillLevelFlags(g);

    const auto legs = truth::reconstructableFromSignal(g);
    CPPUNIT_ASSERT_EQUAL(std::size_t{1}, legs.size());
    CPPUNIT_ASSERT_EQUAL(uint32_t{0}, legs[0]);
    CPPUNIT_ASSERT(g.particles()[0].isAtLevel(truth::LevelFlag::ReconstructableFromSignal));
    CPPUNIT_ASSERT(!g.particles()[1].isAtLevel(truth::LevelFlag::ReconstructableFromSignal));
    CPPUNIT_ASSERT(!g.particles()[2].isAtLevel(truth::LevelFlag::ReconstructableFromSignal));
  }

  // The level table is what every lookup is derived from, so a row added to it must
  // carry its name, its bit and its place in the owned mask without a second edit.
  void testLevelTableIsTheSingleSource() {
    CPPUNIT_ASSERT_EQUAL(truth::kLevelTable.size(), truth::kAllLevels.size());
    uint32_t mask = 0;
    for (auto const& row : truth::kLevelTable) {
      // name and flag round trip through the lookups
      CPPUNIT_ASSERT(row.level == truth::levelFromName(truth::levelName(row.level)));
      CPPUNIT_ASSERT(row.flag == truth::levelFlagOf(row.level));
      CPPUNIT_ASSERT(std::string(row.name) == std::string(truth::levelName(row.level)));
      // every bit is distinct
      CPPUNIT_ASSERT_EQUAL(uint32_t{0}, mask & static_cast<uint32_t>(row.flag));
      mask |= static_cast<uint32_t>(row.flag);
    }
    // the mask fillLevelFlags clears is exactly the table, and never the Signal bit
    CPPUNIT_ASSERT_EQUAL(mask, truth::kOwnedLevelFlags);
    CPPUNIT_ASSERT_EQUAL(uint32_t{0}, truth::kOwnedLevelFlags & static_cast<uint32_t>(truth::LevelFlag::Signal));
    CPPUNIT_ASSERT_THROW(std::ignore = truth::levelFromName("noSuchLevel"), cms::Exception);
  }

  // A re-convergent history: the tau gives a pi0 and a photon, and the pi0 gives the
  // same photon. The walk stops at the pi0 on one path and still reaches the photon on
  // the other, so the level must drop the photon. A level holding both would put a
  // particle and its own parent in one denominator.
  void testReconvergentHistoryStaysAnAntichain() {
    GraphBuilder b(3, 2);
    auto set = [&](uint32_t i, int32_t pdg, int16_t st) {
      auto& d = b.graph.particles()[i];
      d.genNode = 100 + int32_t(i);
      d.pdgId = pdg;
      d.status = st;
      d.momentum = math::XYZTLorentzVectorD(10., 0., 0., 10.);
    };
    set(0, 15, 2);   // tau
    set(1, 111, 2);  // pi0, reconstructable, so the walk stops here
    set(2, 22, 1);   // photon, reached from the tau AND from the pi0
    b.addDecay(0, 0);
    b.addProduction(0, 1);
    b.addProduction(0, 2);
    b.addDecay(1, 1);
    b.addProduction(1, 2);
    truth::Graph g = b.finish();
    g.reconstructablePdgIds() = {111};
    g.particles()[0].setLevel(truth::LevelFlag::Signal);

    const auto legs = truth::levelAntichain(g, truth::Level::ReconstructableFromSignal);
    CPPUNIT_ASSERT_EQUAL(std::size_t{1}, legs.size());
    CPPUNIT_ASSERT_EQUAL(uint32_t{1}, legs[0]);
  }

  // The two levels that hang off the artificial source vertices: Upstream collects the
  // ISR side, UnderlyingEvent the spectators, and each keeps the legs that produced
  // nothing further. A leg's own role vertex decides which level it lands in.
  void testStableLegsFromUpstreamAndUnderlyingEvent() {
    GraphBuilder b(4, 2);
    auto set = [&](uint32_t i, int32_t pdg, int16_t st) {
      auto& d = b.graph.particles()[i];
      d.genNode = 100 + int32_t(i);
      d.pdgId = pdg;
      d.status = st;
      d.momentum = math::XYZTLorentzVectorD(5., 0., 0., 5.);
    };
    set(0, 22, 1);   // ISR photon off the Upstream vertex
    set(1, 211, 1);  // spectator off the UnderlyingEvent vertex
    set(2, 111, 2);  // spectator that decays, so it is not a leg
    set(3, 22, 1);   // its daughter, which is
    b.graph.vertices()[0].role = static_cast<uint8_t>(truth::VertexRole::Upstream);
    b.graph.vertices()[1].role = static_cast<uint8_t>(truth::VertexRole::UnderlyingEvent);
    b.addProduction(0, 0);
    b.addProduction(1, 1);
    b.addProduction(1, 2);
    truth::Graph g = b.finish();

    const auto upstream = truth::levelAntichain(g, truth::Level::StableLegsFromUpstream);
    CPPUNIT_ASSERT_EQUAL(std::size_t{1}, upstream.size());
    CPPUNIT_ASSERT_EQUAL(uint32_t{0}, upstream[0]);

    const auto ue = truth::levelAntichain(g, truth::Level::UnderlyingEvent);
    CPPUNIT_ASSERT_EQUAL(std::size_t{2}, ue.size());
    CPPUNIT_ASSERT_EQUAL(uint32_t{1}, ue[0]);
    CPPUNIT_ASSERT_EQUAL(uint32_t{2}, ue[1]);
  }

  // A diquark carries the flavour digits of the hadron it fragments into and is that
  // hadron's ancestor, so accepting it would cover and drop the real hadron.
  void testDiquarksAreNotHeavyFlavourHadrons() {
    CPPUNIT_ASSERT(!truth::hadronHasQuark(5101, 5));
    CPPUNIT_ASSERT(!truth::hadronHasQuark(5103, 5));
    CPPUNIT_ASSERT(!truth::hadronHasQuark(4101, 4));
    CPPUNIT_ASSERT(!truth::hadronHasQuark(5503, 5));
    CPPUNIT_ASSERT(truth::hadronHasQuark(521, 5));
    CPPUNIT_ASSERT(truth::hadronHasQuark(5122, 5));
    CPPUNIT_ASSERT(truth::hadronHasQuark(421, 4));
    CPPUNIT_ASSERT(truth::hadronHasQuark(4122, 4));
  }

  // Every level answers on a graph with no particles, and stamping one is a no-op.
  void testEmptyGraphStampsNothing() {
    truth::Graph g;
    for (const truth::Level level : truth::kAllLevels) {
      CPPUNIT_ASSERT(truth::levelAntichain(g, level).empty());
    }
    truth::fillLevelFlags(g);
    CPPUNIT_ASSERT_EQUAL(uint32_t{0}, g.nParticles());
  }

  // A well-formed graph reports no cycle, and the walk terminates on a graph where one
  // particle has several parents and several children.
  void testAcyclicGraphReportsNoCycle() {
    truth::Graph decay = buildDecay();
    CPPUNIT_ASSERT(truth::particlesOnCycles(decay).empty());
    truth::Graph tausAndPi0 = buildTausAndPi0();
    CPPUNIT_ASSERT(truth::particlesOnCycles(tausAndPi0).empty());
    truth::Graph empty;
    CPPUNIT_ASSERT(truth::particlesOnCycles(empty).empty());
  }

  // REQUIRED: a directed cycle is reported by id, and every particle on it is named.
  // The levels cannot describe a cycle, so the caller must be able to see one.
  //   p0 -> v0 -> p1 -> v1 -> p2 -> v2 -> p0
  void testCycleIsReported() {
    GraphBuilder b(3, 3);
    for (uint32_t i = 0; i < 3; ++i) {
      auto& p = b.graph.particles()[i];
      p.genNode = 100 + static_cast<int32_t>(i);
      p.pdgId = 211;
      p.status = 2;
      b.addDecay(i, i);
      b.addProduction(i, (i + 1) % 3);
    }
    truth::Graph g = b.finish();

    const std::vector<uint32_t> onCycle = truth::particlesOnCycles(g);
    CPPUNIT_ASSERT_EQUAL(std::size_t{3}, onCycle.size());
    for (uint32_t i = 0; i < 3; ++i) {
      CPPUNIT_ASSERT(std::find(onCycle.begin(), onCycle.end(), i) != onCycle.end());
    }
  }

  // A cycle that hangs off an acyclic chain names only the particles on the cycle, and
  // stamping still runs, so the levels a cycle does not reach stay usable.
  void testCycleDoesNotStopStamping() {
    // p0 (hard process) decays at v0 to p1; p1 -> v1 -> p2 -> v2 -> p1 is the cycle.
    GraphBuilder b(3, 3);
    auto& seed = b.graph.particles()[0];
    seed.genNode = 100;
    seed.pdgId = 6;
    seed.status = 2;
    seed.statusFlags = truth::detail::kIsHardProcess;
    for (uint32_t i = 1; i < 3; ++i) {
      auto& p = b.graph.particles()[i];
      p.genNode = 100 + static_cast<int32_t>(i);
      p.pdgId = 211;
      p.status = 2;
    }
    b.addDecay(0, 0);
    b.addProduction(0, 1);
    b.addDecay(1, 1);
    b.addProduction(1, 2);
    b.addDecay(2, 2);
    b.addProduction(2, 1);
    truth::Graph g = b.finish();

    const std::vector<uint32_t> onCycle = truth::particlesOnCycles(g);
    CPPUNIT_ASSERT_EQUAL(std::size_t{2}, onCycle.size());
    CPPUNIT_ASSERT(std::find(onCycle.begin(), onCycle.end(), 0u) == onCycle.end());

    truth::fillLevelFlags(g);
    CPPUNIT_ASSERT_EQUAL(uint32_t{3}, g.nParticles());
  }

  // A SIM continuation is transport, not decay. The TenTau topology that exposed it:
  // tau -> K0S, where the generator decays the K0S to two pions AND Geant4 interacts
  // it in material, producing neutrons at a SIM vertex. The visible final state is
  // the GEN pions; the nuclear secondaries must never be labelled.
  void testSimContinuationIsNotADecay() {
    GraphBuilder b(6, 3);
    auto set = [&](uint32_t i, int32_t pdg, int16_t st, bool gen) {
      auto& d = b.graph.particles()[i];
      d.genNode = gen ? 100 + int32_t(i) : -1;
      d.simNode = gen ? -1 : 500 + int32_t(i);
      d.pdgId = pdg;
      d.status = st;
      d.momentum = math::XYZTLorentzVectorD(10., 0., 0., 10.);
    };
    set(0, 15, 2, true);     // tau, decays
    set(1, 310, 2, true);    // K0S, GEN-decayed AND SIM-interacted
    set(2, 211, 1, true);    // pi+
    set(3, -211, 1, true);   // pi-
    set(4, 2112, 0, false);  // SIM neutron from the nuclear interaction
    set(5, 2112, 0, false);  // SIM neutron
    b.addDecay(0, 0);        // tau -> K0S (GEN vertex)
    b.addProduction(0, 1);
    b.addDecay(1, 1);  // K0S -> pi pi (GEN vertex)
    b.addProduction(1, 2);
    b.addProduction(1, 3);
    b.addDecay(1, 2);  // K0S SIM interaction vertex
    b.addProduction(2, 4);
    b.addProduction(2, 5);
    truth::Graph g = b.finish();
    g.vertices()[2].genNode = -1;  // the interaction vertex is SIM-only
    g.vertices()[2].simNode = 900;
    g.reconstructablePdgIds() = {111};
    g.particles()[0].setLevel(truth::LevelFlag::Signal);

    const auto legs = truth::reconstructableFromSignal(g);
    CPPUNIT_ASSERT_EQUAL(std::size_t{2}, legs.size());
    CPPUNIT_ASSERT_EQUAL(uint32_t{2}, legs[0]);
    CPPUNIT_ASSERT_EQUAL(uint32_t{3}, legs[1]);
  }

  // REQUIRED: a three-prong tau decay labels the three charged pions, and the
  // intermediate resonance they came from is walked THROUGH, never labelled.
  //   p0 tau (signal) -> v0 -> p1 a1 -> v1 -> p2,p3,p4 pi+/-
  void testThreeProngThroughAnIntermediateResonance() {
    GraphBuilder b(5, 2);
    auto set = [&](uint32_t i, int32_t pdg, int16_t st) {
      auto& d = b.graph.particles()[i];
      d.genNode = 100 + i;
      d.pdgId = pdg;
      d.status = st;
      d.momentum = math::XYZTLorentzVectorD(10., 0., 0., 10.);
    };
    set(0, 15, 2);     // tau
    set(1, 20213, 2);  // a1, an intermediate the detector never sees as an object
    set(2, 211, 1);
    set(3, -211, 1);
    set(4, 211, 1);
    b.addDecay(0, 0);
    b.addProduction(0, 1);
    b.addDecay(1, 1);
    b.addProduction(1, 2);
    b.addProduction(1, 3);
    b.addProduction(1, 4);
    truth::Graph g = b.finish();
    g.reconstructablePdgIds() = {111};
    g.particles()[0].setLevel(truth::LevelFlag::Signal);
    truth::fillLevelFlags(g);

    const auto legs = truth::reconstructableFromSignal(g);
    CPPUNIT_ASSERT_EQUAL(std::size_t{3}, legs.size());
    CPPUNIT_ASSERT(!g.particles()[1].isAtLevel(truth::LevelFlag::ReconstructableFromSignal));
    for (uint32_t id : {2u, 3u, 4u}) {
      CPPUNIT_ASSERT(g.particles()[id].isAtLevel(truth::LevelFlag::ReconstructableFromSignal));
    }

    // Adding the a1 to the configuration labels it instead of its prongs, which is the
    // documented escape hatch.
    g.reconstructablePdgIds() = {111, 20213};
    const auto stopped = truth::reconstructableFromSignal(g);
    CPPUNIT_ASSERT_EQUAL(std::size_t{1}, stopped.size());
    CPPUNIT_ASSERT_EQUAL(uint32_t{1}, stopped[0]);
  }

  // REQUIRED: the jet roots are the hard-scatter partons and nothing else. The top is
  // excluded because its b is deeper, which is also the physics: the top decays before it
  // hadronises. The incoming beam gluon is excluded for the same structural reason, and
  // that matters because it sits at beam rapidity with the whole event below it.
  void testPartonJetsKeepTheQuarksNotTheTopOrTheBeam() {
    truth::Graph g = buildTopDecay();
    const auto jets = truth::partonJets(g);

    const std::vector<uint32_t> expected = {2u, 4u, 5u};  // b, u, dbar
    CPPUNIT_ASSERT_EQUAL(expected.size(), jets.size());
    CPPUNIT_ASSERT(std::is_permutation(expected.begin(), expected.end(), jets.begin()));

    CPPUNIT_ASSERT(std::find(jets.begin(), jets.end(), 0u) == jets.end());  // incoming gluon
    CPPUNIT_ASSERT(std::find(jets.begin(), jets.end(), 1u) == jets.end());  // top
    CPPUNIT_ASSERT(std::find(jets.begin(), jets.end(), 3u) == jets.end());  // W

    // No member may be an ancestor of another, the property every level owes its
    // denominator.
    for (uint32_t id : jets) {
      for (auto const& descendant : truth::Particle(&g, id).descendants()) {
        CPPUNIT_ASSERT(std::find(jets.begin(), jets.end(), descendant.id()) == jets.end());
      }
    }

    truth::fillLevelFlags(g);
    for (uint32_t id : expected) {
      CPPUNIT_ASSERT(g.particles()[id].isAtLevel(truth::LevelFlag::PartonJets));
    }
    CPPUNIT_ASSERT(!g.particles()[1].isAtLevel(truth::LevelFlag::PartonJets));
  }

  // REQUIRED: a hard-process leg that is a lepton is not a jet. The tau fixture is a
  // hard-process particle and the level must still come out empty rather than adopting it.
  void testPartonJetsExcludeLeptons() {
    truth::Graph g = buildDecay();
    CPPUNIT_ASSERT(truth::partonJets(g).empty());
  }

  // REQUIRED: one member per physical heavy-flavour decay, not one per generator copy,
  // and the member is the hadron that DECAYS WEAKLY. A B* radiates down to a B and both
  // are b hadrons; the level keeps the B alone, which is the particle CMS ghost
  // association names and the only one of the two with a displaced decay vertex.
  void testHeavyFlavourKeepsTheWeaklyDecayingHadron() {
    truth::Graph g = buildHeavyFlavour();

    // Both are b hadrons by species, which is what makes the antichain the load-bearing part.
    CPPUNIT_ASSERT(truth::atLevel(g, 1u, truth::Level::BHadrons));
    CPPUNIT_ASSERT(truth::atLevel(g, 2u, truth::Level::BHadrons));

    const auto bees = truth::levelAntichain(g, truth::Level::BHadrons);
    CPPUNIT_ASSERT_EQUAL(std::size_t{1}, bees.size());
    CPPUNIT_ASSERT_EQUAL(uint32_t{2}, bees[0]);
  }

  // REQUIRED: beauty and charm stay separate levels. The D descends from the B, so one
  // combined level would keep the B and drop the D, and charm would silently vanish.
  void testBeautyAndCharmAreSeparateLevels() {
    truth::Graph g = buildHeavyFlavour();

    const auto cees = truth::levelAntichain(g, truth::Level::CHadrons);
    CPPUNIT_ASSERT_EQUAL(std::size_t{1}, cees.size());
    CPPUNIT_ASSERT_EQUAL(uint32_t{3}, cees[0]);

    // The nesting that makes a combined level wrong.
    bool bIsAncestorOfC = false;
    for (auto const& ancestor : truth::Particle(&g, 3u).ancestors()) {
      bIsAncestorOfC = bIsAncestorOfC || ancestor.id() == 1u;
    }
    CPPUNIT_ASSERT(bIsAncestorOfC);

    truth::fillLevelFlags(g);
    CPPUNIT_ASSERT(g.particles()[2].isAtLevel(truth::LevelFlag::BHadrons));
    CPPUNIT_ASSERT(!g.particles()[1].isAtLevel(truth::LevelFlag::BHadrons));
    CPPUNIT_ASSERT(g.particles()[3].isAtLevel(truth::LevelFlag::CHadrons));
    CPPUNIT_ASSERT(!g.particles()[3].isAtLevel(truth::LevelFlag::BHadrons));
  }

  // REQUIRED: one entry per physical hadronically decaying tau. The last copy of a
  // radiative chain is the member; the radiating copy and a leptonically decaying tau
  // are not.
  void testVisibleTauIsTheLastHadronicCopy() {
    truth::Graph g = buildTausAndPi0();
    const auto taus = truth::levelAntichain(g, truth::Level::VisibleTau);
    CPPUNIT_ASSERT_EQUAL(std::size_t{1}, taus.size());
    CPPUNIT_ASSERT_EQUAL(uint32_t{1}, taus[0]);
    truth::fillLevelFlags(g);
    CPPUNIT_ASSERT(g.particles()[1].isAtLevel(truth::LevelFlag::VisibleTau));
    CPPUNIT_ASSERT(!g.particles()[0].isAtLevel(truth::LevelFlag::VisibleTau));
    CPPUNIT_ASSERT(!g.particles()[5].isAtLevel(truth::LevelFlag::VisibleTau));
  }

  // REQUIRED: the event-wide reconstructable final state exists WITHOUT a Signal flag,
  // and a pi0 is one object: the pi0 is a member, its photons are not, and neutrinos
  // are dropped.
  void testReconstructableFinalStateNeedsNoSignal() {
    truth::Graph g = buildTausAndPi0();
    g.reconstructablePdgIds() = {111};
    const auto legs = truth::levelAntichain(g, truth::Level::ReconstructableFinalState);
    CPPUNIT_ASSERT(!legs.empty());
    auto has = [&legs](uint32_t id) { return std::find(legs.begin(), legs.end(), id) != legs.end(); };
    CPPUNIT_ASSERT(has(9));
    CPPUNIT_ASSERT(!has(10));
    CPPUNIT_ASSERT(!has(11));
    CPPUNIT_ASSERT(has(3));
    CPPUNIT_ASSERT(has(2));
    CPPUNIT_ASSERT(!has(4));
    CPPUNIT_ASSERT(!has(7));
    // The radiating tau is walked through, never labelled.
    CPPUNIT_ASSERT(!has(0));
    CPPUNIT_ASSERT(!has(1));
  }
};

CPPUNIT_TEST_SUITE_REGISTRATION(LevelFlags_t);
