// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "DataFormats/HepMCCandidate/interface/GenStatusFlags.h"
#include "PhysicsTools/TruthInfo/interface/GenGraphBuild.h"

namespace {

  constexpr uint16_t kHardProcess = 1u << reco::GenStatusFlags::kIsHardProcess;
  constexpr uint16_t kLastCopy = 1u << reco::GenStatusFlags::kIsLastCopy;

  // A hand-built GEN record with the two things the collapse has to handle, plus a
  // re-convergent branch. Vertex barcodes are negative, particle barcodes positive,
  // as HepMC2 writes them.
  //
  //   V-1: (nothing in)          -> 1 g   status 21, hard process
  //                              -> 2 g   status 21, hard process
  //   V-2: 1,2 in                -> 3 H   status 22, hard process        (first copy)
  //   V-3: 3 in                  -> 4 H   status 44                      (intermediate copy)
  //   V-4: 4 in                  -> 5 H   status 62, last copy           (last copy)
  //   V-5: 5 in                  -> 6 b   status 71                      (shower parton)
  //                              -> 7 bbar status 71                     (shower parton)
  //   V-6: 6,7 in                -> 8 string 92                          (shower object)
  //   V-7: 8 in                  -> 9  pi+ status 1
  //                              -> 10 pi- status 1
  //
  // Kept: 1, 2 (hard process), 3 (hard process), 5 (last copy of a resonance), 9, 10
  // (stable). Collapsed: 4 (intermediate copy), 6, 7 (partons), 8 (string).
  truth::GenBuild buildRecord() {
    truth::GenBuild gb;

    struct Particle {
      int barcode;
      int32_t pdgId;
      int16_t status;
      uint16_t flags;
      int prodVertex;
      int endVertex;  // 0 = none
    };

    const std::vector<Particle> particles = {
        // A beam proton, as HepMC2 writes it: last copy, and NO production vertex.
        {11, 2212, 4, kLastCopy, 0, -1},
        {1, 21, 21, kHardProcess, -1, -2},
        {2, 21, 21, kHardProcess, -1, -2},
        {3, 25, 22, kHardProcess, -2, -3},
        {4, 25, 44, 0, -3, -4},
        {5, 25, 62, kLastCopy, -4, -5},
        {6, 5, 71, kLastCopy, -5, -6},
        {7, -5, 71, kLastCopy, -5, -6},
        {8, 92, 92, kLastCopy, -6, -7},
        {9, 211, 1, kLastCopy, -7, 0},
        {10, -211, 1, kLastCopy, -7, 0},
    };

    for (int vbc = -1; vbc >= -7; --vbc)
      gb.vtxBarcodes.push_back(vbc);

    for (auto const& p : particles) {
      gb.partBarcodes.push_back(p.barcode);
      gb.particleBarcodeByIndex.push_back(p.barcode);
      gb.particlePdgIdByBarcode.emplace(p.barcode, p.pdgId);
      gb.particleStatusByBarcode.emplace(p.barcode, p.status);
      gb.particleStatusFlagsByBarcode.emplace(p.barcode, p.flags);
      if (p.prodVertex != 0)
        gb.vtxToPart.emplace_back(p.prodVertex, p.barcode);
      if (p.endVertex != 0)
        gb.partToVtx.emplace_back(p.barcode, p.endVertex);
    }

    return gb;
  }

  // A minimum-bias style record for the compact form. No status flags: the compact
  // record never reads them.
  //
  //   V-1: 1,2 beam protons in   -> 3 string 92   status 2
  //   V-2: 3 in                  -> 4 pi0  status 2
  //                              -> 5 pi+  status 1
  //                              -> 6 eta  status 2
  //   V-3: 4 in                  -> 7, 8 gamma  status 1
  //   V-4: 6 in                  -> 9 pi0  status 2
  //                              -> 10 gamma status 1
  //   V-5: 9 in                  -> 11 e+, 12 e-, 13 gamma  status 1 (Dalitz)
  truth::GenBuild buildMinBiasRecord() {
    truth::GenBuild gb;

    struct Particle {
      int barcode;
      int32_t pdgId;
      int16_t status;
      int prodVertex;  // 0 = none
      int endVertex;   // 0 = none
    };

    const std::vector<Particle> particles = {
        {1, 2212, 4, 0, -1},
        {2, 2212, 4, 0, -1},
        {3, 92, 2, -1, -2},
        {4, 111, 2, -2, -3},
        {5, 211, 1, -2, 0},
        {6, 221, 2, -2, -4},
        {7, 22, 1, -3, 0},
        {8, 22, 1, -3, 0},
        {9, 111, 2, -4, -5},
        {10, 22, 1, -4, 0},
        {11, -11, 1, -5, 0},
        {12, 11, 1, -5, 0},
        {13, 22, 1, -5, 0},
    };

    for (int vbc = -1; vbc >= -5; --vbc)
      gb.vtxBarcodes.push_back(vbc);

    for (auto const& p : particles) {
      gb.partBarcodes.push_back(p.barcode);
      gb.particleBarcodeByIndex.push_back(p.barcode);
      gb.particlePdgIdByBarcode.emplace(p.barcode, p.pdgId);
      gb.particleStatusByBarcode.emplace(p.barcode, p.status);
      if (p.prodVertex != 0)
        gb.vtxToPart.emplace_back(p.prodVertex, p.barcode);
      if (p.endVertex != 0)
        gb.partToVtx.emplace_back(p.barcode, p.endVertex);
    }

    return gb;
  }

  truth::CompactGenParticle const* findCompact(std::vector<truth::CompactGenParticle> const& record, int barcode) {
    const auto it = std::find_if(
        record.begin(), record.end(), [barcode](truth::CompactGenParticle const& p) { return p.barcode == barcode; });
    return it != record.end() ? &*it : nullptr;
  }

  bool hasVertexToParticle(truth::GenBuild const& gb, int vbc, int pbc) {
    return std::find(gb.vtxToPart.begin(), gb.vtxToPart.end(), std::make_pair(vbc, pbc)) != gb.vtxToPart.end();
  }

  bool hasParticleToVertex(truth::GenBuild const& gb, int pbc, int vbc) {
    return std::find(gb.partToVtx.begin(), gb.partToVtx.end(), std::make_pair(pbc, vbc)) != gb.partToVtx.end();
  }

  bool contains(std::vector<int> const& v, int x) { return std::find(v.begin(), v.end(), x) != v.end(); }

  // Every kept particle must be reachable from the source vertices, which are the ones
  // with no incoming particle: that is where the callers attach the GenEvent node.
  std::unordered_set<int> reachableParticles(truth::GenBuild const& gb) {
    std::unordered_set<int> withIncoming;
    for (auto const& [pbc, vbc] : gb.partToVtx)
      withIncoming.insert(vbc);

    std::vector<int> vertexStack;
    for (int vbc : gb.vtxBarcodes) {
      if (withIncoming.count(vbc) == 0)
        vertexStack.push_back(vbc);
    }

    // Mirror the callers: when no vertex is a source, the GenEvent node is attached to
    // every vertex instead. A collider record always lands here, because the beam
    // particles give the first vertex an incoming particle.
    if (vertexStack.empty())
      vertexStack.assign(gb.vtxBarcodes.begin(), gb.vtxBarcodes.end());

    std::unordered_set<int> seenParticles;
    std::unordered_set<int> seenVertices(vertexStack.begin(), vertexStack.end());
    while (!vertexStack.empty()) {
      const int vbc = vertexStack.back();
      vertexStack.pop_back();
      for (auto const& [from, pbc] : gb.vtxToPart) {
        if (from != vbc || !seenParticles.insert(pbc).second)
          continue;
        for (auto const& [pFrom, to] : gb.partToVtx) {
          if (pFrom == pbc && seenVertices.insert(to).second)
            vertexStack.push_back(to);
        }
      }
    }
    return seenParticles;
  }

}  // namespace

class TestGenGraphBuild : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestGenGraphBuild);
  CPPUNIT_TEST(testKeptSet);
  CPPUNIT_TEST(testContractedAncestry);
  CPPUNIT_TEST(testNoOrphans);
  CPPUNIT_TEST(testSimContinuationKeeps);
  CPPUNIT_TEST(testNoStatusFlagsReportsDegraded);
  CPPUNIT_TEST(testCompactKeepsStableOnly);
  CPPUNIT_TEST(testCompactKeepsListedSpecies);
  CPPUNIT_TEST(testCompactChainsThroughKeptSpecies);
  CPPUNIT_TEST(testCompactEdgeCases);
  CPPUNIT_TEST_SUITE_END();

public:
  void testKeptSet();
  void testContractedAncestry();
  void testNoOrphans();
  void testSimContinuationKeeps();
  void testNoStatusFlagsReportsDegraded();
  void testCompactKeepsStableOnly();
  void testCompactKeepsListedSpecies();
  void testCompactChainsThroughKeptSpecies();
  void testCompactEdgeCases();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestGenGraphBuild);

// The keep rule survives exactly the hard process, the last copy of the resonance and
// the stable particles; the intermediate copy, the shower partons and the string go.
void TestGenGraphBuild::testKeptSet() {
  auto gb = buildRecord();
  // buildRecord() sets packed status flags, so the flag-based keep rules are live.
  CPPUNIT_ASSERT(truth::collapseGenShower(gb, {}));

  const std::vector<int> expected = {1, 2, 3, 5, 9, 10, 11};
  std::vector<int> kept = gb.partBarcodes;
  std::sort(kept.begin(), kept.end());
  CPPUNIT_ASSERT(kept == expected);

  CPPUNIT_ASSERT(gb.particlePdgIdByBarcode.size() == expected.size());
  CPPUNIT_ASSERT(gb.particleStatusByBarcode.size() == expected.size());
  CPPUNIT_ASSERT(gb.particleStatusFlagsByBarcode.size() == expected.size());
  CPPUNIT_ASSERT(gb.particleBarcodeByIndex.size() == expected.size());

  // A vertex survives only if it still produces a survivor: V-6 produced the string
  // only, so it is bypassed.
  CPPUNIT_ASSERT(contains(gb.vtxBarcodes, -1));
  CPPUNIT_ASSERT(contains(gb.vtxBarcodes, -2));
  CPPUNIT_ASSERT(contains(gb.vtxBarcodes, -4));
  CPPUNIT_ASSERT(contains(gb.vtxBarcodes, -7));
  CPPUNIT_ASSERT(!contains(gb.vtxBarcodes, -3));
  CPPUNIT_ASSERT(!contains(gb.vtxBarcodes, -6));
}

// Ancestry is contracted, not cut: the resonance last copy hangs off the hard-process
// copy, and the stable particles hang off the last copy, through real vertices.
void TestGenGraphBuild::testContractedAncestry() {
  auto gb = buildRecord();
  // buildRecord() sets packed status flags, so the flag-based keep rules are live.
  CPPUNIT_ASSERT(truth::collapseGenShower(gb, {}));

  // 3 (hard process H) -> V-4 -> 5 (last copy H), the intermediate copy 4 gone.
  CPPUNIT_ASSERT(hasParticleToVertex(gb, 3, -4));
  CPPUNIT_ASSERT(hasVertexToParticle(gb, -4, 5));

  // 5 -> V-7 -> 9, 10, the two partons and the string gone.
  CPPUNIT_ASSERT(hasParticleToVertex(gb, 5, -7));
  CPPUNIT_ASSERT(hasVertexToParticle(gb, -7, 9));
  CPPUNIT_ASSERT(hasVertexToParticle(gb, -7, 10));

  // The two incoming partons still meet at the hard vertex.
  CPPUNIT_ASSERT(hasParticleToVertex(gb, 1, -2));
  CPPUNIT_ASSERT(hasParticleToVertex(gb, 2, -2));
  CPPUNIT_ASSERT(hasVertexToParticle(gb, -2, 3));

  // No edge may refer to a particle or a vertex that is gone.
  for (auto const& [vbc, pbc] : gb.vtxToPart) {
    CPPUNIT_ASSERT(contains(gb.vtxBarcodes, vbc));
    CPPUNIT_ASSERT(contains(gb.partBarcodes, pbc));
  }
  for (auto const& [pbc, vbc] : gb.partToVtx) {
    CPPUNIT_ASSERT(contains(gb.partBarcodes, pbc));
    CPPUNIT_ASSERT(contains(gb.vtxBarcodes, vbc));
  }
}

// The contraction must not COST reachability: a survivor that the source vertices could
// reach before must still be reachable after. It cannot be asked to do better than the
// record it is given. A beam particle has no production vertex in the HepMC record, so
// nothing points at it and it is unreachable before the collapse as well; measured on
// ttbar as exactly 2 unreachable GenParticles per event both before and after.
void TestGenGraphBuild::testNoOrphans() {
  auto before = buildRecord();
  const auto reachableBefore = reachableParticles(before);

  auto after = buildRecord();
  CPPUNIT_ASSERT(truth::collapseGenShower(after, {}));
  const auto reachableAfter = reachableParticles(after);

  for (int pbc : after.partBarcodes) {
    if (reachableBefore.count(pbc) != 0) {
      CPPUNIT_ASSERT(reachableAfter.count(pbc) != 0);
    }
  }

  // Everything the collapse kept and could reach is reachable, so the only survivors
  // outside the reachable set are the ones the record itself never attached.
  for (int pbc : after.partBarcodes) {
    if (reachableAfter.count(pbc) == 0) {
      CPPUNIT_ASSERT(reachableBefore.count(pbc) == 0);
    }
  }

  // The beam particle of buildRecord() is exactly that case, so this test would pass
  // vacuously if it ever stopped being kept.
  CPPUNIT_ASSERT(contains(after.partBarcodes, 11));
  CPPUNIT_ASSERT(reachableBefore.count(11) == 0);
}

// A SimTrack continuing a particle keeps it whatever its status and flags say, and the
// contraction then routes through it.
void TestGenGraphBuild::testSimContinuationKeeps() {
  auto gb = buildRecord();
  CPPUNIT_ASSERT(truth::collapseGenShower(gb, std::unordered_set<int>{6}));

  CPPUNIT_ASSERT(contains(gb.partBarcodes, 6));
  CPPUNIT_ASSERT(!contains(gb.partBarcodes, 7));

  CPPUNIT_ASSERT(hasParticleToVertex(gb, 5, -5));
  CPPUNIT_ASSERT(hasVertexToParticle(gb, -5, 6));

  // 9 and 10 now have two nearest surviving ancestors, 6 and the last copy 5, because
  // only one of the two partons survived.
  CPPUNIT_ASSERT(hasParticleToVertex(gb, 6, -7));
  CPPUNIT_ASSERT(hasParticleToVertex(gb, 5, -7));

  // Every survivor is reachable except the beam particle, which the record itself leaves
  // without a production vertex. See testNoOrphans.
  const auto reachable = reachableParticles(gb);
  for (int pbc : gb.partBarcodes) {
    CPPUNIT_ASSERT((reachable.count(pbc) != 0) == (pbc != 11));
  }
}

// Without packed status flags the isHardProcess and isLastCopy rules are dead: the
// collapse must SAY so, and the keep set degrades to SIM-continued plus status 1.
void TestGenGraphBuild::testNoStatusFlagsReportsDegraded() {
  auto gb = buildRecord();
  gb.particleStatusFlagsByBarcode.clear();

  CPPUNIT_ASSERT(!truth::collapseGenShower(gb, std::unordered_set<int>{6}));

  const std::vector<int> expected = {6, 9, 10};
  std::vector<int> kept = gb.partBarcodes;
  std::sort(kept.begin(), kept.end());
  CPPUNIT_ASSERT(kept == expected);
}

// With no listed species the compact record is the stable particles, every one on the
// vertex of the interaction.
void TestGenGraphBuild::testCompactKeepsStableOnly() {
  const auto record = truth::compactGen(buildMinBiasRecord(), {});

  std::vector<int> kept;
  for (auto const& p : record) {
    kept.push_back(p.barcode);
    CPPUNIT_ASSERT_EQUAL(int16_t{1}, p.status);
    CPPUNIT_ASSERT_EQUAL(0, p.parent);
    CPPUNIT_ASSERT_EQUAL(0, p.decayVertex);
  }
  std::sort(kept.begin(), kept.end());
  const std::vector<int> expected = {5, 7, 8, 10, 11, 12, 13};
  CPPUNIT_ASSERT(kept == expected);
}

// A listed species survives with its decay vertex, and its decay products hang on it.
// An unlisted parent is walked through: the photon of the eta hangs on the interaction.
void TestGenGraphBuild::testCompactKeepsListedSpecies() {
  const auto record = truth::compactGen(buildMinBiasRecord(), {111});

  CPPUNIT_ASSERT_EQUAL(std::size_t{9}, record.size());
  CPPUNIT_ASSERT(findCompact(record, 3) == nullptr);
  CPPUNIT_ASSERT(findCompact(record, 6) == nullptr);

  auto const* pi0 = findCompact(record, 4);
  CPPUNIT_ASSERT(pi0 != nullptr);
  CPPUNIT_ASSERT_EQUAL(int16_t{2}, pi0->status);
  CPPUNIT_ASSERT_EQUAL(-3, pi0->decayVertex);
  // Made by the string, so it hangs on the interaction.
  CPPUNIT_ASSERT_EQUAL(0, pi0->parent);

  CPPUNIT_ASSERT_EQUAL(4, findCompact(record, 7)->parent);
  CPPUNIT_ASSERT_EQUAL(4, findCompact(record, 8)->parent);
  CPPUNIT_ASSERT_EQUAL(0, findCompact(record, 5)->parent);
  CPPUNIT_ASSERT_EQUAL(0, findCompact(record, 10)->parent);

  // The Dalitz pi0 under the unlisted eta: it hangs on the interaction, its products on it.
  CPPUNIT_ASSERT_EQUAL(0, findCompact(record, 9)->parent);
  for (int product : {11, 12, 13})
    CPPUNIT_ASSERT_EQUAL(9, findCompact(record, product)->parent);
}

// Listing the parent species too gives a chain: eta, then pi0, then the Dalitz products.
void TestGenGraphBuild::testCompactChainsThroughKeptSpecies() {
  const auto record = truth::compactGen(buildMinBiasRecord(), {111, 221});

  auto const* eta = findCompact(record, 6);
  CPPUNIT_ASSERT(eta != nullptr);
  CPPUNIT_ASSERT_EQUAL(-4, eta->decayVertex);
  CPPUNIT_ASSERT_EQUAL(0, eta->parent);
  CPPUNIT_ASSERT_EQUAL(6, findCompact(record, 9)->parent);
  CPPUNIT_ASSERT_EQUAL(6, findCompact(record, 10)->parent);
  CPPUNIT_ASSERT_EQUAL(9, findCompact(record, 13)->parent);
}

// A listed species survives only as what it is in the record: stable with no decay vertex,
// or decaying with one; with no end vertex it is dropped. Only a vertex it alone enters
// makes it a parent. The ancestor walk terminates on a cycle.
void TestGenGraphBuild::testCompactEdgeCases() {
  truth::GenBuild gb;
  auto add = [&gb](int barcode, int32_t pdgId, int16_t status, int prodVertex, int endVertex) {
    gb.partBarcodes.push_back(barcode);
    gb.particlePdgIdByBarcode.emplace(barcode, pdgId);
    gb.particleStatusByBarcode.emplace(barcode, status);
    if (prodVertex != 0)
      gb.vtxToPart.emplace_back(prodVertex, barcode);
    if (endVertex != 0)
      gb.partToVtx.emplace_back(barcode, endVertex);
  };
  //   1 pi0 status 1, no end vertex
  //   2 pi0 status 2, no end vertex
  //   3 pi0 status 2 and 4 pi+ status 2 both enter V-1 -> 5 gamma
  //   V-2: 6 in -> 7 eta, 8 gamma ; V-3: 7 in -> 6 eta   (a cycle through unlisted 6, 7)
  add(1, 111, 1, 0, 0);
  add(2, 111, 2, 0, 0);
  add(3, 111, 2, 0, -1);
  add(4, 211, 2, 0, -1);
  add(5, 22, 1, -1, 0);
  add(6, 221, 2, -3, -2);
  add(7, 221, 2, -2, -3);
  add(8, 22, 1, -2, 0);
  for (int vbc = -1; vbc >= -3; --vbc)
    gb.vtxBarcodes.push_back(vbc);

  const auto record = truth::compactGen(gb, {111});

  auto const* stablePi0 = findCompact(record, 1);
  CPPUNIT_ASSERT(stablePi0 != nullptr);
  CPPUNIT_ASSERT_EQUAL(0, stablePi0->decayVertex);
  CPPUNIT_ASSERT(findCompact(record, 2) == nullptr);

  auto const* sharedPi0 = findCompact(record, 3);
  CPPUNIT_ASSERT(sharedPi0 != nullptr);
  CPPUNIT_ASSERT_EQUAL(-1, sharedPi0->decayVertex);
  CPPUNIT_ASSERT_EQUAL(0, findCompact(record, 5)->parent);

  CPPUNIT_ASSERT_EQUAL(0, findCompact(record, 8)->parent);
  CPPUNIT_ASSERT_EQUAL(std::size_t{4}, record.size());
}
