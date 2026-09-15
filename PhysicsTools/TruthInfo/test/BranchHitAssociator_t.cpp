// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <cmath>
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
    b.addHit(truth::HitChannel::Calo, 0, 100, 10, 1.0f, 0);
    b.addHit(truth::HitChannel::Calo, 0, 100, 11, 1.0f, 0);
    b.addHit(truth::HitChannel::Calo, 0, 101, 11, 1.0f, 0);
    b.addHit(truth::HitChannel::Calo, 0, 101, 12, 2.0f, 0);
    return b.finish();
  }

  // Same topology populated on the *tracker* channel (cells 20,21,22), plus one
  // calo cell (10) that the tracker associator must ignore.
  truth::LogicalGraphHitIndex buildTrackerIndex() {
    truth::LogicalGraphHitIndexBuilder b(2);
    b.setSimTrackForParticle(0, 0, 100);
    b.setSimTrackForParticle(1, 0, 101);
    b.addParticleChild(0, 1);
    b.addHit(truth::HitChannel::Calo, 0, 100, 10, 1.0f, 0);  // calo channel
    b.addHit(truth::HitChannel::Tracker, 0, 100, 20, 1.0f);
    b.addHit(truth::HitChannel::Tracker, 0, 100, 21, 1.0f);
    b.addHit(truth::HitChannel::Tracker, 0, 101, 21, 1.0f);
    b.addHit(truth::HitChannel::Tracker, 0, 101, 22, 2.0f);
    return b.finish();
  }

  // Two independent roots sharing one cell, with UNEQUAL energies so the arithmetic
  // below has no accidental symmetry:
  //   p0 direct: cell 10 (e3)
  //   p1 direct: cell 10 (e1), cell 11 (e4)
  // => cellTotal = {10:4, 11:4}, so on cell 10 the sim fractions are 0.75 and 0.25.
  truth::LogicalGraphHitIndex buildScoreIndex() {
    truth::LogicalGraphHitIndexBuilder b(2);
    b.setSimTrackForParticle(0, 0, 100);
    b.setSimTrackForParticle(1, 0, 101);
    b.addHit(truth::HitChannel::Calo, 0, 100, 10, 3.0f, 0);
    b.addHit(truth::HitChannel::Calo, 0, 101, 10, 1.0f, 0);
    b.addHit(truth::HitChannel::Calo, 0, 101, 11, 4.0f, 0);
    return b.finish();
  }

  // One branch depositing in TWO detectors of the calorimetric channel, as a real one
  // does: cell 10 of detector 8 (an endcap detector) with energy 2, cell 1 of detector
  // 3 (a barrel one) with energy 8. Its channel energy is 10, its detector-8 energy 2.
  constexpr uint32_t kEndcapCell = (8u << 28) | 10u;
  constexpr uint32_t kBarrelCell = (3u << 28) | 1u;

  truth::LogicalGraphHitIndex buildTwoDetectorIndex() {
    truth::LogicalGraphHitIndexBuilder b(1);
    b.setSimTrackForParticle(0, 0, 100);
    b.addHit(truth::HitChannel::Calo, 0, 100, kEndcapCell, 2.0f, 0);
    b.addHit(truth::HitChannel::Calo, 0, 100, kBarrelCell, 8.0f, 0);
    return b.finish();
  }

  // Two particles crossing ONE module, told apart by the cell: p0 fires cells 5 and 6,
  // p1 fires cell 7. Keyed by module alone, every reco hit on module 10 would match both.
  truth::LogicalGraphHitIndex buildCellIndex() {
    truth::LogicalGraphHitIndexBuilder b(2);
    b.setCellKeyed(truth::HitChannel::Tracker, true);
    b.setSimTrackForParticle(0, 0, 100);
    b.setSimTrackForParticle(1, 0, 101);
    b.addHit(truth::HitChannel::Tracker, 0, 100, 10, 1.0f, 5);
    b.addHit(truth::HitChannel::Tracker, 0, 100, 10, 1.0f, 6);
    // The same cell twice: one entry, as a module-keyed index coalesces a repeated detId.
    b.addHit(truth::HitChannel::Tracker, 0, 100, 10, 1.0f, 5);
    b.addHit(truth::HitChannel::Tracker, 0, 101, 10, 1.0f, 7);
    return b.finish();
  }

  // A parent and its child crossing one module on different cells: p0 fires cell 5 of
  // module 10 and cell 5 of module 20, its child p1 fires cell 7 of module 10. The
  // parent's subgraph therefore owns both cells of module 10.
  truth::LogicalGraphHitIndex buildAncestorCellIndex() {
    truth::LogicalGraphHitIndexBuilder b(2);
    b.setCellKeyed(truth::HitChannel::Tracker, true);
    b.setSimTrackForParticle(0, 0, 100);
    b.setSimTrackForParticle(1, 0, 101);
    b.addParticleChild(0, 1);
    b.addHit(truth::HitChannel::Tracker, 0, 100, 10, 1.0f, 5);
    b.addHit(truth::HitChannel::Tracker, 0, 100, 20, 1.0f, 5);
    b.addHit(truth::HitChannel::Tracker, 0, 101, 10, 1.0f, 7);
    return b.finish();
  }

  // One particle on one module, keyed by module: the truth says "somewhere in module 10"
  // and names no cell.
  truth::LogicalGraphHitIndex buildModuleKeyedIndex() {
    truth::LogicalGraphHitIndexBuilder b(1);
    b.setSimTrackForParticle(0, 0, 100);
    b.addHit(truth::HitChannel::Tracker, 0, 100, 10, 1.0f);
    return b.finish();
  }

  truth::BranchHitAssociator trackerAssociator(truth::LogicalGraphHitIndex const& index) {
    return truth::BranchHitAssociator(
        index, {}, truth::BranchHitAssociator::Metric::SharedHits, truth::HitChannel::Tracker);
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
  CPPUNIT_TEST(testTiclScoreArithmetic);
  CPPUNIT_TEST(testSharedEnergyFractionCountsOnlyTheRequestedDetectors);
  CPPUNIT_TEST(testZeroFractionObjectScoresWorst);
  CPPUNIT_TEST(testReverseScoreNeverExceedsOne);
  CPPUNIT_TEST(testRecHitEnergyTableWeightsCells);
  CPPUNIT_TEST(testEqualScoresRankTheTightestBranchFirst);
  CPPUNIT_TEST(testAdaptivePicksTheObjectiveMinimum);
  CPPUNIT_TEST(testAdaptiveCeilingKeepsTheClimbLow);
  CPPUNIT_TEST(testAdaptiveFallsBackWhenTheCeilingRejectsEverything);
  CPPUNIT_TEST(testAdaptiveReturnsNoMatchOnAnEmptyCandidateList);
  CPPUNIT_TEST(testCellsSeparateTwoParticlesOnOneModule);
  CPPUNIT_TEST(testAModuleWithoutACellMatchesAnyCellOfIt);
  CPPUNIT_TEST(testAnAncestorKeepsEveryCellOfAModule);
  CPPUNIT_TEST(testAModuleKeyedBranchIsNeverOverCounted);
  CPPUNIT_TEST_SUITE_END();

public:
  void testSharedEnergyBestBranch();
  void testSharedHitsMetric();
  void testGenericRecoObjectInterface();
  void testTrackerChannel();
  void testEmptyRootsMatchNothingWhenRestricted();
  void testReverseScoreIsBranchNormalized();
  void testTiclScoreArithmetic();
  void testSharedEnergyFractionCountsOnlyTheRequestedDetectors();
  void testZeroFractionObjectScoresWorst();
  void testReverseScoreNeverExceedsOne();
  void testRecHitEnergyTableWeightsCells();
  void testEqualScoresRankTheTightestBranchFirst();
  void testAdaptivePicksTheObjectiveMinimum();
  void testAdaptiveCeilingKeepsTheClimbLow();
  void testAdaptiveFallsBackWhenTheCeilingRejectsEverything();
  void testAdaptiveReturnsNoMatchOnAnEmptyCandidateList();
  void testCellsSeparateTwoParticlesOnOneModule();
  void testAModuleWithoutACellMatchesAnyCellOfIt();
  void testAnAncestorKeepsEveryCellOfAModule();
  void testAModuleKeyedBranchIsNeverOverCounted();
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
      index, {}, truth::BranchHitAssociator::Metric::SharedHits, truth::HitChannel::Calo);
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
                                        truth::HitChannel::Calo,
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

void TestBranchHitAssociator::testTiclScoreArithmetic() {
  auto index = buildScoreIndex();
  truth::BranchHitAssociator assoc(index);  // SharedEnergy, all roots

  // Reco object: half of cell 10 and all of cell 11. RecoHit::energy is deliberately
  // absurd, because the shared-energy metric must take the cell energy from the index.
  std::vector<truth::RecoHit> reco{{10, 99.f, 0.5f}, {11, 99.f, 1.0f}};
  auto matches = assoc.bestBranches(reco);
  CPPUNIT_ASSERT_EQUAL(std::size_t(2), matches.size());

  // Hand-computed, following AllTracksterToSimTracksterAssociatorsByHitsProducer.cc
  // :341-364 (recoToSim) and :428-453 (simToReco), with the cell energy as the rechit
  // energy. Reco energies: 0.5*4 = 2 on cell 10, 1.0*4 = 4 on cell 11.
  // recoToSim denominator = 2^2 + 4^2 = 20.
  const truth::BranchMatch* root0 = nullptr;
  const truth::BranchMatch* root1 = nullptr;
  for (auto const& m : matches) {
    (m.rootParticleId == 0 ? root0 : root1) = &m;
  }
  CPPUNIT_ASSERT(root0 != nullptr && root1 != nullptr);

  // Root 1 owns 1 on cell 10 and 4 on cell 11.
  //   recoToSim: max(0, 2-1)^2 + max(0, 4-4)^2 = 1, over 20.
  //   shared energy: min(2,1) + min(4,4) = 5, and the branch owns 5, so the fraction
  //   is 1 while the reco object is only half of cell 10: sim-normalised, as intended.
  //   simToReco: the reco object covers every unit of the branch, so 0. That the reco
  //   object has MORE energy than the branch on cell 10 is a good association.
  CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0 / 20.0, root1->score, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(5.0, root1->sharedEnergy, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, root1->sharedEnergyFraction, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, root1->reverseScore, 1e-6);

  // Root 0 owns 3 on cell 10 and nothing on cell 11.
  //   recoToSim: max(0, 2-3)^2 + max(0, 4-0)^2 = 16, over 20.
  //   shared energy: min(2,3) + min(4,0) = 2, over the branch's own 3.
  //   simToReco: max(0, 3-2)^2 = 1, over the branch self energy 3^2 = 9.
  CPPUNIT_ASSERT_DOUBLES_EQUAL(16.0 / 20.0, root0->score, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, root0->sharedEnergy, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0 / 3.0, root0->sharedEnergyFraction, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0 / 9.0, root0->reverseScore, 1e-6);

  // The shared energy fraction is NOT one minus the score in either direction: the
  // scores are squared and energy weighted, the fraction is linear. That is exactly why
  // HGCalValidator gates efficiency on the fraction and purity on the score.
  CPPUNIT_ASSERT(std::abs((1.f - root0->reverseScore) - root0->sharedEnergyFraction) > 0.2f);
}

namespace {
  // One reco object covered by three levels of the same chain, as the merge-join yields
  // them: the leaf covers little of the object and nothing else, the parent covers it
  // well, the grandparent covers it entirely and spreads far beyond it.
  truth::BranchMatch match(uint32_t root, float score, float reverseScore) {
    truth::BranchMatch m;
    m.rootParticleId = root;
    m.score = score;
    m.reverseScore = reverseScore;
    return m;
  }

  std::vector<truth::BranchMatch> chainOfThree() {
    return {match(2, 0.4f, 0.0f), match(1, 0.1f, 0.2f), match(0, 0.0f, 0.9f)};
  }
}  // namespace

void TestBranchHitAssociator::testAdaptivePicksTheObjectiveMinimum() {
  const auto matches = chainOfThree();
  // Objectives at weight 1: 0.40, 0.30, 0.90. The parent is the climb's answer.
  const auto best = truth::BranchHitAssociator::bestAdaptiveBranch(matches, 1.f, 1.f);
  CPPUNIT_ASSERT_EQUAL(uint32_t(1), best.rootParticleId);

  // Weight 0 is the Fixed point's objective: the widest branch always covers the most.
  const auto lowest = truth::BranchHitAssociator::bestAdaptiveBranch(matches, 0.f, 1.f);
  CPPUNIT_ASSERT_EQUAL(uint32_t(0), lowest.rootParticleId);
}

void TestBranchHitAssociator::testAdaptiveCeilingKeepsTheClimbLow() {
  const auto matches = chainOfThree();
  // A ceiling of 0.1 admits only the leaf, whose objective is the worst of the three.
  const auto best = truth::BranchHitAssociator::bestAdaptiveBranch(matches, 1.f, 0.1f);
  CPPUNIT_ASSERT_EQUAL(uint32_t(2), best.rootParticleId);
}

void TestBranchHitAssociator::testAdaptiveFallsBackWhenTheCeilingRejectsEverything() {
  // Every candidate spreads beyond the reco object, so the ceiling admits none. A reco
  // object that shares hits must still get an answer, so the unconstrained minimum wins.
  const std::vector<truth::BranchMatch> matches{match(5, 0.6f, 0.5f), match(4, 0.2f, 0.7f)};
  const auto best = truth::BranchHitAssociator::bestAdaptiveBranch(matches, 1.f, 0.1f);
  CPPUNIT_ASSERT_EQUAL(uint32_t(4), best.rootParticleId);
}

void TestBranchHitAssociator::testAdaptiveReturnsNoMatchOnAnEmptyCandidateList() {
  // What a caller that filtered every candidate away hands in. No match is the honest
  // answer; the fallback must not invent one.
  const std::vector<truth::BranchMatch> none;
  const auto best = truth::BranchHitAssociator::bestAdaptiveBranch(none, 1.f, 1.f);
  CPPUNIT_ASSERT_EQUAL(truth::BranchMatch::kInvalidRoot, best.rootParticleId);
}

void TestBranchHitAssociator::testEqualScoresRankTheTightestBranchFirst() {
  // buildIndex: particle 0 is the parent of particle 1, so its subgraph holds every
  // cell of 1 (11 and 12) and more (10). By shared cells a reco object on cells 11 and
  // 12 is covered by both with score 0; the child is the tighter branch and must come
  // first, even though the parent has the lower id.
  auto index = buildIndex();
  truth::BranchHitAssociator assoc(index, {}, truth::BranchHitAssociator::Metric::SharedHits, truth::HitChannel::Calo);
  std::vector<truth::RecoHit> reco{{11, 0.f, 1.0f}, {12, 0.f, 1.0f}};
  auto matches = assoc.bestBranches(reco);
  CPPUNIT_ASSERT_EQUAL(std::size_t(2), matches.size());
  CPPUNIT_ASSERT_DOUBLES_EQUAL(matches[0].score, matches[1].score, 1e-6);
  CPPUNIT_ASSERT_EQUAL(uint32_t(1), matches[0].rootParticleId);
  CPPUNIT_ASSERT_EQUAL(uint32_t(0), matches[1].rootParticleId);
  CPPUNIT_ASSERT(matches[0].reverseScore < matches[1].reverseScore);
}

void TestBranchHitAssociator::testRecHitEnergyTableWeightsCells() {
  auto index = buildScoreIndex();  // cell 10: p0 3, p1 1 (total 4); cell 11: p1 4

  // Reco object: half of cell 10, all of cell 11, and all of cell 13, which no
  // particle touched (a noise rechit).
  std::vector<truth::RecoHit> reco{{10, 0.f, 0.5f}, {11, 0.f, 1.0f}, {13, 0.f, 1.0f}};

  // Without a table the noise cell weighs nothing and the scores are those of
  // testTiclScoreArithmetic.
  truth::BranchHitAssociator simWeighted(index);
  auto simMatches = simWeighted.bestBranches(reco);
  CPPUNIT_ASSERT_EQUAL(std::size_t(2), simMatches.size());
  for (auto const& m : simMatches)
    CPPUNIT_ASSERT_DOUBLES_EQUAL(m.rootParticleId == 1 ? 1.0 / 20.0 : 16.0 / 20.0, m.score, 1e-6);

  // Rechit energies: cell 10 reads 8 (twice its sim total), cell 11 has no rechit,
  // cell 13 reads 5. Added out of order and with a repeated cell to exercise finalize.
  truth::CellEnergyTable table;
  table.add(13, 5.f);
  table.add(10, 6.f);
  table.add(10, 2.f);
  table.finalize();
  CPPUNIT_ASSERT_EQUAL(std::size_t(2), table.size());
  CPPUNIT_ASSERT_DOUBLES_EQUAL(8.0, table.energy(10), 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, table.energy(11), 1e-6);

  truth::BranchHitAssociator assoc(index,
                                   {},
                                   truth::BranchHitAssociator::Metric::SharedEnergy,
                                   truth::HitChannel::Calo,
                                   true,
                                   truth::BranchHitAssociator::kAllDetectors,
                                   &table);
  auto matches = assoc.bestBranches(reco);
  CPPUNIT_ASSERT_EQUAL(std::size_t(2), matches.size());
  const truth::BranchMatch* root0 = nullptr;
  const truth::BranchMatch* root1 = nullptr;
  for (auto const& m : matches)
    (m.rootParticleId == 0 ? root0 : root1) = &m;
  CPPUNIT_ASSERT(root0 != nullptr && root1 != nullptr);

  // Reco energies: 0.5 * 8 = 4 on cell 10, 1 * 0 = 0 on cell 11, 1 * 5 = 5 on cell 13.
  // recoToSim denominator = 16 + 0 + 25 = 41. The noise cell is paid for in full.
  //
  // Root 1 owns 1/4 of cell 10, so 2 of its rechit energy, and all of cell 11, so 0.
  //   recoToSim: max(0, 4-2)^2 + 0 + 5^2 = 29, over 41.
  //   shared: min(4,2) = 2, over the branch's own 2: fraction 1.
  //   simToReco: self energy 2^2 = 4, all captured: 0.
  CPPUNIT_ASSERT_DOUBLES_EQUAL(29.0 / 41.0, root1->score, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, root1->sharedEnergy, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, root1->sharedEnergyFraction, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, root1->reverseScore, 1e-6);

  // Root 0 owns 3/4 of cell 10, so 6, and nothing else.
  //   recoToSim: max(0, 4-6)^2 + 0 + 25 = 25, over 41.
  //   shared: min(4,6) = 4, over the branch's own 6.
  //   simToReco: max(0, 6-4)^2 = 4, over 6^2 = 36.
  CPPUNIT_ASSERT_DOUBLES_EQUAL(25.0 / 41.0, root0->score, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, root0->sharedEnergy, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0 / 6.0, root0->sharedEnergyFraction, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0 / 36.0, root0->reverseScore, 1e-6);

  // An empty table is the same as no table.
  truth::CellEnergyTable empty;
  truth::BranchHitAssociator fallback(index,
                                      {},
                                      truth::BranchHitAssociator::Metric::SharedEnergy,
                                      truth::HitChannel::Calo,
                                      true,
                                      truth::BranchHitAssociator::kAllDetectors,
                                      &empty);
  auto fallbackMatches = fallback.bestBranches(reco);
  for (auto const& m : fallbackMatches)
    CPPUNIT_ASSERT_DOUBLES_EQUAL(m.rootParticleId == 1 ? 1.0 / 20.0 : 16.0 / 20.0, m.score, 1e-6);
}

void TestBranchHitAssociator::testSharedEnergyFractionCountsOnlyTheRequestedDetectors() {
  auto index = buildTwoDetectorIndex();

  // A reco object that takes the whole endcap cell and nothing else, which is what an
  // endcap reco collection can do at best for this branch.
  std::vector<truth::RecoHit> reco{{kEndcapCell, 99.f, 1.0f}};

  // Whole channel in the denominator: shared 2 over the branch's 2 + 8, which is below
  // the 0.5 efficiency gate although the reco object took everything it could reach.
  truth::BranchHitAssociator wholeChannel(index);
  auto wholeMatches = wholeChannel.bestBranches(reco);
  CPPUNIT_ASSERT_EQUAL(std::size_t(1), wholeMatches.size());
  CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, wholeMatches.front().sharedEnergy, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0 / 10.0, wholeMatches.front().sharedEnergyFraction, 1e-6);

  // Detector 8 only: shared 2 over the branch's 2 there, so the fraction is 1.
  truth::BranchHitAssociator endcapOnly(index,
                                        {},
                                        truth::BranchHitAssociator::Metric::SharedEnergy,
                                        truth::HitChannel::Calo,
                                        /*emptyRootsMeansAll=*/true,
                                        truth::BranchHitAssociator::detectorBit(kEndcapCell));
  auto endcapMatches = endcapOnly.bestBranches(reco);
  CPPUNIT_ASSERT_EQUAL(std::size_t(1), endcapMatches.size());
  CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, endcapMatches.front().sharedEnergy, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, endcapMatches.front().sharedEnergyFraction, 1e-6);

  // The restriction touches the fraction and nothing else: both TICL scores keep their
  // own denominators over the whole channel. recoToSim is 0 (the reco object covers its
  // own energy exactly, 1.0 * 2 against the branch's 2); simToReco is the barrel energy
  // the reco object misses, (2^2 + 8^2 - 2^2) / (2^2 + 8^2) = 64 / 68.
  CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, endcapMatches.front().score, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(wholeMatches.front().score, endcapMatches.front().score, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(64.0 / 68.0, endcapMatches.front().reverseScore, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(wholeMatches.front().reverseScore, endcapMatches.front().reverseScore, 1e-6);

  // A mixed object covering both cells against the endcap-only denominator: the
  // fraction numerator counts the same detectors as its denominator, so the fraction
  // is 1, not (2 + 8) / 2. sharedEnergy keeps the whole channel.
  std::vector<truth::RecoHit> mixed{{kBarrelCell, 99.f, 1.0f}, {kEndcapCell, 99.f, 1.0f}};
  auto mixedMatches = endcapOnly.bestBranches(mixed);
  CPPUNIT_ASSERT_EQUAL(std::size_t(1), mixedMatches.size());
  CPPUNIT_ASSERT_DOUBLES_EQUAL(10.0, mixedMatches.front().sharedEnergy, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, mixedMatches.front().sharedEnergyFraction, 1e-6);
}

void TestBranchHitAssociator::testZeroFractionObjectScoresWorst() {
  auto index = buildIndex();
  truth::BranchHitAssociator assoc(index);  // SharedEnergy, all roots

  // Every hit of the object carries fraction 0, so its self-energy denominator is
  // 0. Such an object must score 1 (worst) on every candidate, never 0 (best).
  std::vector<truth::RecoHit> reco{{10, 1.0f, 0.0f}, {11, 2.0f, 0.0f}, {12, 2.0f, 0.0f}};
  auto matches = assoc.bestBranches(reco);

  CPPUNIT_ASSERT(!matches.empty());
  for (auto const& m : matches) {
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, m.score, 1e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, m.sharedEnergy, 1e-6);
  }
}

void TestBranchHitAssociator::testReverseScoreNeverExceedsOne() {
  // The reverse score is the fraction of the branch's own energy the reco object fails
  // to cover, so it lies in [0, 1]. A working point whose ceiling is 1 or more therefore
  // rejects nothing and repeats the unconstrained point. The ladder in the cff depends
  // on this bound, so pin it here.
  auto index = buildScoreIndex();
  truth::BranchHitAssociator assoc(index);
  const std::vector<std::vector<truth::RecoHit>> probes = {{{10, 4.0f, 1.0f}},
                                                           {{10, 4.0f, 0.25f}},
                                                           {{11, 4.0f, 1.0f}},
                                                           {{10, 4.0f, 1.0f}, {11, 4.0f, 1.0f}},
                                                           {{10, 4.0f, 0.0f}},
                                                           {{10, 4.0f, 4.0f}}};
  for (auto const& probe : probes) {
    for (auto const& m : assoc.bestBranches(probe)) {
      CPPUNIT_ASSERT(m.reverseScore >= 0.f);
      CPPUNIT_ASSERT(m.reverseScore <= 1.f);
    }
  }
}

void TestBranchHitAssociator::testCellsSeparateTwoParticlesOnOneModule() {
  auto index = buildCellIndex();
  // The repeated cell coalesced, so particle 0 owns two cells of the module and not three.
  CPPUNIT_ASSERT_EQUAL(std::size_t(2), index.directHits(truth::HitChannel::Tracker, 0).size());
  CPPUNIT_ASSERT_EQUAL(std::size_t(1), index.directHits(truth::HitChannel::Tracker, 1).size());

  auto assoc = trackerAssociator(index);
  // A cluster on cells 5 and 6 of module 10 belongs to particle 0 alone. Keyed by module
  // this object would have matched particle 1 just as well.
  std::vector<truth::RecoHit> reco{{10, 1.f, 1.f, 5}, {10, 1.f, 1.f, 6}};
  auto matches = assoc.bestBranches(reco);
  CPPUNIT_ASSERT_EQUAL(std::size_t(1), matches.size());
  CPPUNIT_ASSERT_EQUAL(uint32_t(0), matches.front().rootParticleId);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, matches.front().score, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, matches.front().reverseScore, 1e-6);

  // The other particle's cell picks the other particle.
  std::vector<truth::RecoHit> other{{10, 1.f, 1.f, 7}};
  auto otherMatches = assoc.bestBranches(other);
  CPPUNIT_ASSERT_EQUAL(std::size_t(1), otherMatches.size());
  CPPUNIT_ASSERT_EQUAL(uint32_t(1), otherMatches.front().rootParticleId);
}

void TestBranchHitAssociator::testAModuleWithoutACellMatchesAnyCellOfIt() {
  auto index = buildCellIndex();
  auto assoc = trackerAssociator(index);
  // An adapter that cannot reach the clusters names the module alone. The answer is then
  // the coarse one, both particles, rather than nothing at all.
  std::vector<truth::RecoHit> reco{{10, 1.f, 1.f}};
  auto matches = assoc.bestBranches(reco);
  CPPUNIT_ASSERT_EQUAL(std::size_t(2), matches.size());
  CPPUNIT_ASSERT_DOUBLES_EQUAL(matches[0].score, matches[1].score, 1e-6);
  // Equal scores rank the tightest branch first: particle 1 owns one cell, particle 0 two.
  CPPUNIT_ASSERT_EQUAL(uint32_t(1), matches[0].rootParticleId);
}

void TestBranchHitAssociator::testAnAncestorKeepsEveryCellOfAModule() {
  auto index = buildAncestorCellIndex();
  auto assoc = trackerAssociator(index);

  // Two rechits: one on the cell the child fired, one on the cell only the parent fired.
  // The parent's subgraph holds both, so it owns the object entirely. Coalescing that
  // subgraph by module alone would drop cell 7 from it and score the parent 0.5.
  std::vector<truth::RecoHit> reco{{10, 1.f, 1.f, 7}, {20, 1.f, 1.f, 5}};
  auto matches = assoc.bestBranches(reco);
  CPPUNIT_ASSERT_EQUAL(std::size_t(2), matches.size());
  CPPUNIT_ASSERT_EQUAL(uint32_t(0), matches.front().rootParticleId);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, matches.front().score, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, matches.front().sharedEnergy, 1e-6);

  // The child owns one of the two rechits.
  CPPUNIT_ASSERT_EQUAL(uint32_t(1), matches.back().rootParticleId);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, matches.back().score, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, matches.back().reverseScore, 1e-6);
}

void TestBranchHitAssociator::testAModuleKeyedBranchIsNeverOverCounted() {
  auto index = buildModuleKeyedIndex();
  auto assoc = trackerAssociator(index);

  // One rechit of three cells against a branch that names the module and no cell. The
  // object is one rechit, the branch is one entry, so both normalisations give 1.
  std::vector<truth::RecoHit> reco{{10, 1.f, 1.f, 5}, {10, 1.f, 1.f, 6}, {10, 1.f, 1.f, 7}};
  auto matches = assoc.bestBranches(reco);
  CPPUNIT_ASSERT_EQUAL(std::size_t(1), matches.size());
  CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, matches.front().score, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, matches.front().reverseScore, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, matches.front().sharedEnergyFraction, 1e-6);
  CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, matches.front().sharedEnergy, 1e-6);
}
