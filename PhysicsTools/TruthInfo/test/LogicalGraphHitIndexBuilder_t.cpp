// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <cstdint>

#include "SimDataFormats/TruthInfo/interface/LogicalGraphHitIndex.h"
#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndexBuilder.h"

// These tests lock in the layout property the Branch view relies on: a particle's
// subgraph hits are a single contiguous std::span, sorted by detId, deduplicated
// by detId with energy accumulated across the whole subtree. That makes a
// Subtree branch's hits == subgraphHits(truth::HitChannel::HGCalCalo, root) with zero gather, and orders them
// for merge-join matching against reco objects.
class TestLogicalGraphHitIndexBuilder : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestLogicalGraphHitIndexBuilder);
  CPPUNIT_TEST(testSubgraphHitsAreSortedContiguousAndAccumulated);
  CPPUNIT_TEST(testDirectHitsAreSortedByDetId);
  CPPUNIT_TEST_SUITE_END();

public:
  void testSubgraphHitsAreSortedContiguousAndAccumulated();
  void testDirectHitsAreSortedByDetId();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestLogicalGraphHitIndexBuilder);

void TestLogicalGraphHitIndexBuilder::testSubgraphHitsAreSortedContiguousAndAccumulated() {
  // particle 0 (track 100) -> child particle 1 (track 101)
  truth::LogicalGraphHitIndexBuilder builder(2);
  builder.setSimTrackForParticle(0, 100);
  builder.setSimTrackForParticle(1, 101);
  builder.addParticleChild(0, 1);

  builder.addHit(truth::HitChannel::HGCalCalo, 100, /*detId=*/10, /*energy=*/1.0f, /*recHitIndex=*/0);
  builder.addHit(truth::HitChannel::HGCalCalo, 100, /*detId=*/5, /*energy=*/2.0f, /*recHitIndex=*/1);
  builder.addHit(
      truth::HitChannel::HGCalCalo, 101, /*detId=*/10, /*energy=*/3.0f, /*recHitIndex=*/0);  // same detId as parent
  builder.addHit(truth::HitChannel::HGCalCalo, 101, /*detId=*/20, /*energy=*/1.5f, /*recHitIndex=*/2);

  auto index = builder.finish();

  auto sub = index.subgraphHits(truth::HitChannel::HGCalCalo, 0);
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
  auto subChild = index.subgraphHits(truth::HitChannel::HGCalCalo, 1);
  CPPUNIT_ASSERT_EQUAL(std::size_t(2), subChild.size());
  CPPUNIT_ASSERT_EQUAL(uint32_t(10), subChild[0].detId);
  CPPUNIT_ASSERT_EQUAL(uint32_t(20), subChild[1].detId);
}

void TestLogicalGraphHitIndexBuilder::testDirectHitsAreSortedByDetId() {
  truth::LogicalGraphHitIndexBuilder builder(1);
  builder.setSimTrackForParticle(0, 7);
  builder.addHit(truth::HitChannel::HGCalCalo, 7, 30, 1.0f, 0);
  builder.addHit(truth::HitChannel::HGCalCalo, 7, 3, 1.0f, 1);
  builder.addHit(truth::HitChannel::HGCalCalo, 7, 17, 1.0f, 2);

  auto index = builder.finish();
  auto direct = index.directHits(truth::HitChannel::HGCalCalo, 0);
  CPPUNIT_ASSERT_EQUAL(std::size_t(3), direct.size());
  CPPUNIT_ASSERT_EQUAL(uint32_t(3), direct[0].detId);
  CPPUNIT_ASSERT_EQUAL(uint32_t(17), direct[1].detId);
  CPPUNIT_ASSERT_EQUAL(uint32_t(30), direct[2].detId);
}
