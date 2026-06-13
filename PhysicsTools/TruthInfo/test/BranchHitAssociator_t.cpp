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
    b.setSimTrackForParticle(0, 100);
    b.setSimTrackForParticle(1, 101);
    b.addParticleChild(0, 1);
    b.addHitForTrack(100, 10, 0, 1.0f);
    b.addHitForTrack(100, 11, 0, 1.0f);
    b.addHitForTrack(101, 11, 0, 1.0f);
    b.addHitForTrack(101, 12, 0, 2.0f);
    return b.finish();
  }

}  // namespace

class TestBranchHitAssociator : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestBranchHitAssociator);
  CPPUNIT_TEST(testSharedEnergyBestBranch);
  CPPUNIT_TEST(testSharedHitsMetric);
  CPPUNIT_TEST(testGenericRecoObjectInterface);
  CPPUNIT_TEST_SUITE_END();

public:
  void testSharedEnergyBestBranch();
  void testSharedHitsMetric();
  void testGenericRecoObjectInterface();
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
