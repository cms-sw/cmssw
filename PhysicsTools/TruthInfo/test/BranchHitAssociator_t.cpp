// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <cstdint>
#include <vector>

#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"
#include "PhysicsTools/TruthInfo/interface/LogicalGraphHitIndexBuilder.h"

namespace {

  // A minimal user reco object: it only has to expose truthHits().
  struct MyRecoObject {
    std::vector<truth::RecoHit> hits;
    [[nodiscard]] const std::vector<truth::RecoHit>& truthHits() const { return hits; }
  };

  // particle 0 (track 100) -> child particle 1 (track 101)
  //   p0 direct: cell10 (e1), cell11 (e1)
  //   p1 direct: cell11 (e1), cell12 (e2)
  // => subgraph(0) = {10:1, 11:2, 12:2}; subgraph(1) = {11:1, 12:2}
  //    cellTotal   = {10:1, 11:2, 12:2}
  truth::LogicalGraphHitIndex buildIndex() {
    truth::LogicalGraphHitIndexBuilder b(2);
    b.setSimTrackForParticle(0, 0, 100);
    b.setSimTrackForParticle(1, 0, 101);
    b.addParticleChild(0, 1);
    b.addHit(truth::HitChannel::HGCalCalo, 0, 100, 10, 1.0f, 0);
    b.addHit(truth::HitChannel::HGCalCalo, 0, 100, 11, 1.0f, 0);
    b.addHit(truth::HitChannel::HGCalCalo, 0, 101, 11, 1.0f, 0);
    b.addHit(truth::HitChannel::HGCalCalo, 0, 101, 12, 2.0f, 0);
    return b.finish();
  }

  // Same topology populated on the *tracker* channel (cells 20,21,22), plus one
  // calo cell (10) that the tracker associator must ignore.
  truth::LogicalGraphHitIndex buildTrackerIndex() {
    truth::LogicalGraphHitIndexBuilder b(2);
    b.setSimTrackForParticle(0, 0, 100);
    b.setSimTrackForParticle(1, 0, 101);
    b.addParticleChild(0, 1);
    b.addHit(truth::HitChannel::HGCalCalo, 0, 100, 10, 1.0f, 0);  // calo channel
    b.addHit(truth::HitChannel::Tracker, 0, 100, 20, 1.0f);
    b.addHit(truth::HitChannel::Tracker, 0, 100, 21, 1.0f);
    b.addHit(truth::HitChannel::Tracker, 0, 101, 21, 1.0f);
    b.addHit(truth::HitChannel::Tracker, 0, 101, 22, 2.0f);
    return b.finish();
  }

}  // namespace

class TestBranchHitAssociator : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestBranchHitAssociator);
  CPPUNIT_TEST(testSharedEnergyBestBranch);
  CPPUNIT_TEST(testSharedHitsMetric);
  CPPUNIT_TEST(testGenericRecoObjectInterface);
  CPPUNIT_TEST(testTrackerChannel);
  CPPUNIT_TEST(testEmptyRootsMatchNothingWhenRestricted);
  CPPUNIT_TEST(testReverseScoreIsBranchNormalized);
  CPPUNIT_TEST_SUITE_END();

public:
  void testSharedEnergyBestBranch();
  void testSharedHitsMetric();
  void testGenericRecoObjectInterface();
  void testTrackerChannel();
  void testEmptyRootsMatchNothingWhenRestricted();
  void testReverseScoreIsBranchNormalized();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestBranchHitAssociator);

void TestBranchHitAssociator::testSharedEnergyBestBranch() {
  auto index = buildIndex();
  truth::BranchHitAssociator assoc(index);  // SharedEnergy, all roots

  // A reco object that perfectly covers cells 10,11,12 (fraction 1).
  std::vector<truth::RecoHit> reco{{10, 1.0f, 1.0f}, {11, 2.0f, 1.0f}, {12, 2.0f, 1.0f}};
  auto matches = assoc.bestBranches(reco);

  CPPUNIT_ASSERT(!matches.empty());
  // Root 0's subtree covers every cell with fraction 1 -> perfect match (score 0).
  CPPUNIT_ASSERT_EQUAL(uint32_t(0), matches.front().rootParticleId);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, matches.front().score, 1e-6);

  // Root 1 covers cell 11 (frac 0.5) and 12 (frac 1) but not 10 -> worse score.
  bool foundRoot1 = false;
  for (auto const& m : matches)
    if (m.rootParticleId == 1) {
      foundRoot1 = true;
      CPPUNIT_ASSERT(m.score > matches.front().score);
    }
  CPPUNIT_ASSERT(foundRoot1);
}

void TestBranchHitAssociator::testSharedHitsMetric() {
  auto index = buildIndex();
  truth::BranchHitAssociator assoc(index, {}, truth::BranchHitAssociator::Metric::SharedHits);

  std::vector<truth::RecoHit> reco{{10, 1.0f, 1.0f}, {11, 1.0f, 1.0f}, {12, 1.0f, 1.0f}};
  auto matches = assoc.bestBranches(reco, /*maxResults=*/1);

  CPPUNIT_ASSERT_EQUAL(std::size_t(1), matches.size());
  // Root 0 shares all 3 cells -> best (score 0); sharedEnergy field carries the count.
  CPPUNIT_ASSERT_EQUAL(uint32_t(0), matches.front().rootParticleId);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, matches.front().sharedEnergy, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, matches.front().score, 1e-6);
}

void TestBranchHitAssociator::testGenericRecoObjectInterface() {
  auto index = buildIndex();
  truth::BranchHitAssociator assoc(index);

  // The generic interface: any object with truthHits() works.
  MyRecoObject obj;
  obj.hits = {{11, 2.0f, 1.0f}, {12, 2.0f, 1.0f}};
  auto matches = assoc.bestBranches(obj);

  CPPUNIT_ASSERT(!matches.empty());
  // Root 0's subtree covers cells 11,12 with fraction 1 (reco->branch perfect),
  // so it is the best match; root 1 only partially covers cell 11.
  CPPUNIT_ASSERT_EQUAL(uint32_t(0), matches.front().rootParticleId);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, matches.front().score, 1e-6);
  CPPUNIT_ASSERT(matches.size() >= 2);  // both root 0 and root 1 are candidates
}

void TestBranchHitAssociator::testTrackerChannel() {
  auto index = buildTrackerIndex();
  truth::BranchHitAssociator assoc(
      index, {}, truth::BranchHitAssociator::Metric::SharedHits, truth::HitChannel::Tracker);

  // Tracker cells 20,21,22 are fully covered by root 0's tracker subgraph.
  std::vector<truth::RecoHit> reco{{20, 1.0f, 1.0f}, {21, 1.0f, 1.0f}, {22, 1.0f, 1.0f}};
  auto matches = assoc.bestBranches(reco, /*maxResults=*/1);
  CPPUNIT_ASSERT_EQUAL(std::size_t(1), matches.size());
  CPPUNIT_ASSERT_EQUAL(uint32_t(0), matches.front().rootParticleId);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, matches.front().sharedEnergy, 1e-6);

  // Channel separation: the tracker associator ignores the calo-only cell (10)...
  std::vector<truth::RecoHit> caloReco{{10, 1.0f, 1.0f}};
  CPPUNIT_ASSERT(assoc.bestBranches(caloReco).empty());
  // ...and a calo associator ignores the tracker cells.
  truth::BranchHitAssociator caloAssoc(
      index, {}, truth::BranchHitAssociator::Metric::SharedHits, truth::HitChannel::HGCalCalo);
  CPPUNIT_ASSERT(caloAssoc.bestBranches(reco).empty());
}

void TestBranchHitAssociator::testEmptyRootsMatchNothingWhenRestricted() {
  auto index = buildIndex();

  // Empty roots with emptyRootsMeansAll=false => no candidate branches, so even a
  // perfectly-covering reco object matches nothing. (Regression: a configured
  // pdg-id restriction that selects no particle in an event must not silently fall
  // back to matching every branch.)
  truth::BranchHitAssociator restricted(index,
                                        {},
                                        truth::BranchHitAssociator::Metric::SharedEnergy,
                                        truth::HitChannel::HGCalCalo,
                                        /*emptyRootsMeansAll=*/false);
  std::vector<truth::RecoHit> reco{{10, 1.0f, 1.0f}, {11, 2.0f, 1.0f}, {12, 2.0f, 1.0f}};
  CPPUNIT_ASSERT(restricted.bestBranches(reco).empty());

  // Sanity: the default (empty roots => all) still matches the same object.
  truth::BranchHitAssociator all(index);
  CPPUNIT_ASSERT(!all.bestBranches(reco).empty());
}

void TestBranchHitAssociator::testReverseScoreIsBranchNormalized() {
  auto index = buildIndex();
  truth::BranchHitAssociator assoc(index);

  // Reco object fully covers cells 10,11,12. Root 1's branch is only {11,12},
  // which the reco object fully contains.
  std::vector<truth::RecoHit> reco{{10, 1.0f, 1.0f}, {11, 2.0f, 1.0f}, {12, 2.0f, 1.0f}};
  auto matches = assoc.bestBranches(reco);

  bool sawRoot1 = false;
  for (auto const& m : matches) {
    if (m.rootParticleId == 0) {
      // Root 0's subtree == the reco object: perfect both ways.
      CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, m.score, 1e-6);
      CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, m.reverseScore, 1e-6);
    } else if (m.rootParticleId == 1) {
      sawRoot1 = true;
      // Reco-centric: the reco object also hits cell 10, which root 1 does not
      // explain -> score > 0. Branch-centric: the reco object covers all of root
      // 1's branch -> reverseScore == 0. This asymmetry is the point of the fix.
      CPPUNIT_ASSERT(m.score > 0.f);
      CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, m.reverseScore, 1e-6);
    }
  }
  CPPUNIT_ASSERT(sawRoot1);
}
