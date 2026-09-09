// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <cstdint>

#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include <map>
#include <vector>

#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndexBuilder.h"

// These tests lock in the layout property the Branch view relies on: a particle's
// subgraph hits are a single contiguous std::span, sorted by detId, deduplicated
// by detId with energy accumulated across the whole subtree. That makes a
// Subtree branch's hits == subgraphHits(truth::HitChannel::Calo, root) with zero gather, and orders them
// for merge-join matching against reco objects.
class TestLogicalGraphHitIndexBuilder : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestLogicalGraphHitIndexBuilder);
  CPPUNIT_TEST(testSubgraphHitsAreSortedContiguousAndAccumulated);
  CPPUNIT_TEST(testDirectHitsAreSortedByDetId);
  CPPUNIT_TEST(testSubgraphDiamondCountsSharedDescendantOnce);
  CPPUNIT_TEST(testSharedStoreKeepsEachHitOnce);
  CPPUNIT_TEST(testSharedStoreFallsBackWhenNotAForest);
  CPPUNIT_TEST(testSharedStoreFallsBackAcrossAGenOnlyChild);
  CPPUNIT_TEST(testSharedStoreFallsBackAcrossAGenOnlyCycle);
  CPPUNIT_TEST_SUITE_END();

public:
  void testSubgraphHitsAreSortedContiguousAndAccumulated();
  void testDirectHitsAreSortedByDetId();
  void testSubgraphDiamondCountsSharedDescendantOnce();
  void testSharedStoreKeepsEachHitOnce();
  void testSharedStoreFallsBackWhenNotAForest();
  void testSharedStoreFallsBackAcrossAGenOnlyChild();
  void testSharedStoreFallsBackAcrossAGenOnlyCycle();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestLogicalGraphHitIndexBuilder);

void TestLogicalGraphHitIndexBuilder::testSubgraphHitsAreSortedContiguousAndAccumulated() {
  // particle 0 (track 100) -> child particle 1 (track 101)
  truth::LogicalGraphHitIndexBuilder builder(2, /*sharedSubgraphStore=*/false);
  builder.setSimTrackForParticle(0, 0, 100);
  builder.setSimTrackForParticle(1, 0, 101);
  builder.addParticleChild(0, 1);

  builder.addHit(truth::HitChannel::Calo, 0, 100, /*detId=*/10, /*energy=*/1.0f, /*recHitIndex=*/0);
  builder.addHit(truth::HitChannel::Calo, 0, 100, /*detId=*/5, /*energy=*/2.0f, /*recHitIndex=*/1);
  builder.addHit(
      truth::HitChannel::Calo, 0, 101, /*detId=*/10, /*energy=*/3.0f, /*recHitIndex=*/0);  // same detId as parent
  builder.addHit(truth::HitChannel::Calo, 0, 101, /*detId=*/20, /*energy=*/1.5f, /*recHitIndex=*/2);

  auto index = builder.finish();

  auto sub = index.subgraphHits(truth::HitChannel::Calo, 0);
  // subtree of 0 = {5, 10, 20}, with detId 10 accumulated across parent+child.
  CPPUNIT_ASSERT_EQUAL(std::size_t(3), sub.size());
  CPPUNIT_ASSERT_EQUAL(uint32_t(5), sub[0].detId);
  CPPUNIT_ASSERT_EQUAL(uint32_t(10), sub[1].detId);
  CPPUNIT_ASSERT_EQUAL(uint32_t(20), sub[2].detId);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, sub[1].energy, 1e-6);  // 1.0 (parent) + 3.0 (child)

  // sorted strictly ascending by detId (merge-join ready).
  for (std::size_t i = 1; i < sub.size(); ++i)
    CPPUNIT_ASSERT(sub[i - 1].detId < sub[i].detId);

  // child subtree is just its own hits.
  auto subChild = index.subgraphHits(truth::HitChannel::Calo, 1);
  CPPUNIT_ASSERT_EQUAL(std::size_t(2), subChild.size());
  CPPUNIT_ASSERT_EQUAL(uint32_t(10), subChild[0].detId);
  CPPUNIT_ASSERT_EQUAL(uint32_t(20), subChild[1].detId);
}

void TestLogicalGraphHitIndexBuilder::testDirectHitsAreSortedByDetId() {
  truth::LogicalGraphHitIndexBuilder builder(1, /*sharedSubgraphStore=*/false);
  builder.setSimTrackForParticle(0, 0, 7);
  builder.addHit(truth::HitChannel::Calo, 0, 7, 30, 1.0f, 0);
  builder.addHit(truth::HitChannel::Calo, 0, 7, 3, 1.0f, 1);
  builder.addHit(truth::HitChannel::Calo, 0, 7, 17, 1.0f, 2);

  auto index = builder.finish();
  auto direct = index.directHits(truth::HitChannel::Calo, 0);
  CPPUNIT_ASSERT_EQUAL(std::size_t(3), direct.size());
  CPPUNIT_ASSERT_EQUAL(uint32_t(3), direct[0].detId);
  CPPUNIT_ASSERT_EQUAL(uint32_t(17), direct[1].detId);
  CPPUNIT_ASSERT_EQUAL(uint32_t(30), direct[2].detId);
}

void TestLogicalGraphHitIndexBuilder::testSubgraphDiamondCountsSharedDescendantOnce() {
  // Re-convergent DAG: 0 -> 1 -> 3 and 0 -> 2 -> 3. Particle 3 is a descendant of
  // 0 along two distinct paths; its hit must contribute to subgraphHits(0) exactly
  // once. (Regression: the old recursive child-subgraph merge summed it once per
  // path, since coalesce() sums equal detIds, doubling the energy.)
  truth::LogicalGraphHitIndexBuilder builder(4, /*sharedSubgraphStore=*/false);
  builder.setSimTrackForParticle(0, 0, 100);
  builder.setSimTrackForParticle(1, 0, 101);
  builder.setSimTrackForParticle(2, 0, 102);
  builder.setSimTrackForParticle(3, 0, 103);
  builder.addParticleChild(0, 1);
  builder.addParticleChild(0, 2);
  builder.addParticleChild(1, 3);
  builder.addParticleChild(2, 3);

  builder.addHit(truth::HitChannel::Calo, 0, 103, /*detId=*/50, /*energy=*/2.0f, /*recHitIndex=*/0);

  auto index = builder.finish();

  auto sub = index.subgraphHits(truth::HitChannel::Calo, 0);
  CPPUNIT_ASSERT_EQUAL(std::size_t(1), sub.size());
  CPPUNIT_ASSERT_EQUAL(uint32_t(50), sub[0].detId);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, sub[0].energy, 1e-6);  // counted once, not 4.0
}

// The shared layout stores every hit exactly once and answers a subgraph as a range of
// that one store, so the range carries the same per-cell energy as the materialised
// aggregate once the caller sums the entries that share a detId.
void TestLogicalGraphHitIndexBuilder::testSharedStoreKeepsEachHitOnce() {
  auto build = [](bool shared) {
    truth::LogicalGraphHitIndexBuilder builder(2, shared);
    builder.setSimTrackForParticle(0, 0, 100);
    builder.setSimTrackForParticle(1, 0, 101);
    builder.addParticleChild(0, 1);
    builder.addHit(truth::HitChannel::Calo, 0, 100, 10, 1.0f, 0);
    builder.addHit(truth::HitChannel::Calo, 0, 100, 5, 2.0f, 1);
    builder.addHit(truth::HitChannel::Calo, 0, 101, 10, 3.0f, 0);
    builder.addHit(truth::HitChannel::Calo, 0, 101, 20, 1.5f, 2);
    return builder.finish();
  };

  const auto materialised = build(false);
  const auto shared = build(true);

  CPPUNIT_ASSERT(!materialised.sharedSubgraphStore());
  CPPUNIT_ASSERT(shared.sharedSubgraphStore());

  // Four hits went in. The materialised layout also keeps a second, aggregated copy;
  // the shared one keeps no duplicate storage at all.
  auto const& sharedChannel = shared.channel(truth::HitChannel::Calo);
  CPPUNIT_ASSERT_EQUAL(std::size_t(4), sharedChannel.directHits.size());
  CPPUNIT_ASSERT(sharedChannel.subgraphHits.empty());
  CPPUNIT_ASSERT(!materialised.channel(truth::HitChannel::Calo).subgraphHits.empty());

  // Same physics content for the root's subgraph: detId 10 sums parent and child.
  std::vector<truth::LogicalGraphHitIndex::Hit> hits;
  shared.appendSubgraphHits(truth::HitChannel::Calo, 0, hits);
  std::map<uint32_t, float> summed;
  for (auto const& hit : hits)
    summed[hit.detId] += hit.energy;

  CPPUNIT_ASSERT_EQUAL(std::size_t(3), summed.size());
  CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, summed[5], 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, summed[10], 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(1.5, summed[20], 1e-6);
}

// A hit-carrying particle with two hit-carrying parents cannot be one contiguous run
// under either of them, so finish() must fall back rather than answer a short subgraph.
void TestLogicalGraphHitIndexBuilder::testSharedStoreFallsBackWhenNotAForest() {
  truth::LogicalGraphHitIndexBuilder builder(4, /*sharedSubgraphStore=*/true);
  builder.setSimTrackForParticle(0, 0, 100);
  builder.setSimTrackForParticle(1, 0, 101);
  builder.setSimTrackForParticle(2, 0, 102);
  builder.setSimTrackForParticle(3, 0, 103);
  builder.addParticleChild(0, 1);
  builder.addParticleChild(0, 2);
  builder.addParticleChild(1, 3);
  builder.addParticleChild(2, 3);
  builder.addHit(truth::HitChannel::Calo, 0, 103, 50, 2.0f, 0);

  const auto index = builder.finish();

  CPPUNIT_ASSERT(!builder.usedSharedStore());
  CPPUNIT_ASSERT(!index.sharedSubgraphStore());

  // Both parents of 3 still see its hit, which is what the fallback buys.
  for (const uint32_t root : {0u, 1u, 2u}) {
    auto sub = index.subgraphHits(truth::HitChannel::Calo, root);
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), sub.size());
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, sub[0].energy, 1e-6);
  }
}

// A hit-carrying particle whose child carries no hits but whose grandchild does. The tree
// is built from hasSimTrack-to-hasSimTrack edges, so the grandchild would land outside the
// grandparent's subtree run and its hits would go missing from that subgraph. finish()
// must fall back rather than answer short.
void TestLogicalGraphHitIndexBuilder::testSharedStoreFallsBackAcrossAGenOnlyChild() {
  // REQUIRED: a SIM to GEN-only to SIM sandwich stays on the SHARED store. This is a
  // decay in flight, and central heavy-ion events carry it at scale: 71 bridges in one
  // Hydjet event. Falling back to the materialised layout there duplicated every hit
  // per ancestor and produced an index above ROOT's 1 GiB single-object limit
  // (cms-sw/cmssw#51638). The bridge is walked through, so the grandparent's subgraph
  // still sees everything below it and the GEN-only node keeps its own union view.
  truth::LogicalGraphHitIndexBuilder builder(3, /*sharedSubgraphStore=*/true);
  builder.setSimTrackForParticle(0, 0, 100);
  // particle 1 is GEN-only: no SimTrack, so it never carries hits itself.
  builder.setSimTrackForParticle(2, 0, 102);
  builder.addParticleChild(0, 1);
  builder.addParticleChild(1, 2);

  builder.addHit(truth::HitChannel::Calo, 0, 100, 10, 1.0f, 0);
  builder.addHit(truth::HitChannel::Calo, 0, 102, 20, 2.0f, 1);

  const auto index = builder.finish();

  CPPUNIT_ASSERT(builder.usedSharedStore());
  CPPUNIT_ASSERT(index.sharedSubgraphStore());

  // The grandparent sees both cells: its subtree run walked THROUGH the bridge.
  {
    auto sub = index.subgraphHits(truth::HitChannel::Calo, 0);
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), sub.size());
    CPPUNIT_ASSERT_EQUAL(uint32_t(10), sub[0].detId);
    CPPUNIT_ASSERT_EQUAL(uint32_t(20), sub[1].detId);
  }
  // The GEN-only bridge sees exactly its own descendant's cell.
  {
    auto sub = index.subgraphHits(truth::HitChannel::Calo, 1);
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), sub.size());
    CPPUNIT_ASSERT_EQUAL(uint32_t(20), sub[0].detId);
  }
  // And the leaf sees itself alone.
  {
    auto sub = index.subgraphHits(truth::HitChannel::Calo, 2);
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), sub.size());
    CPPUNIT_ASSERT_EQUAL(uint32_t(20), sub[0].detId);
  }
}

// A GEN-only CYCLE between the hit-carrying parent and a hit-carrying descendant. The
// closure walk must not memoize "reaches nothing" for a cycle member whose exit leads to
// a SimTrack: a child still being computed counts as reaching, so the guard fires and the
// builder falls back rather than answer a subgraph short of the descendant's hits.
void TestLogicalGraphHitIndexBuilder::testSharedStoreFallsBackAcrossAGenOnlyCycle() {
  truth::LogicalGraphHitIndexBuilder builder(4, /*sharedSubgraphStore=*/true);
  builder.setSimTrackForParticle(0, 0, 100);
  // particles 1 and 2 are GEN-only and form a cycle; the exit from 1 reaches particle 3.
  builder.setSimTrackForParticle(3, 0, 103);
  builder.addParticleChild(0, 2);
  builder.addParticleChild(1, 2);
  builder.addParticleChild(1, 3);
  builder.addParticleChild(2, 1);

  builder.addHit(truth::HitChannel::Calo, 0, 100, 10, 1.0f, 0);
  builder.addHit(truth::HitChannel::Calo, 0, 103, 20, 2.0f, 1);

  const auto index = builder.finish();

  // The bridge walk visits each GEN-only node once, so a cycle terminates and the SIM
  // exit is attached under the one SIM parent: representable, so the SHARED store holds.
  CPPUNIT_ASSERT(builder.usedSharedStore());
  CPPUNIT_ASSERT(index.sharedSubgraphStore());

  // The parent still sees the descendant's cell through the cycle, which was always the
  // required behaviour; only the layout that provides it changed.
  auto sub = index.subgraphHits(truth::HitChannel::Calo, 0);
  CPPUNIT_ASSERT_EQUAL(std::size_t(2), sub.size());
  CPPUNIT_ASSERT_EQUAL(uint32_t(10), sub[0].detId);
  CPPUNIT_ASSERT_EQUAL(uint32_t(20), sub[1].detId);
}
