// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "PhysicsTools/TruthInfo/interface/RecoHitAdapters.h"

namespace {
  reco::CaloCluster layerCluster(std::vector<std::pair<uint32_t, float>> const& hits) {
    reco::CaloCluster lc;
    for (auto const& [detId, fraction] : hits)
      lc.addHitAndFraction(DetId(detId), fraction);
    return lc;
  }

  ticl::Trackster trackster(std::vector<unsigned int> const& layerClusters) {
    ticl::Trackster t;
    for (const unsigned int lc : layerClusters) {
      t.vertices().push_back(lc);
      t.vertex_multiplicity().push_back(1.f);
    }
    return t;
  }

  TICLCandidate candidate(std::vector<edm::Ptr<ticl::Trackster>> const& tracksters) {
    TICLCandidate c;
    c.setTracksters(tracksters);
    return c;
  }
}  // namespace

class TestRecoHitAdapters : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestRecoHitAdapters);
  CPPUNIT_TEST(candidateWithoutTracksterHasNoHit);
  CPPUNIT_TEST(nullTracksterAddsNoHit);
  CPPUNIT_TEST(oneTracksterGivesItsHits);
  CPPUNIT_TEST(sharedCellIsCoalesced);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {
    // Cell 30 is in layer clusters 0 and 1, which belong to two tracksters.
    layerClusters_ = {layerCluster({{20, 1.f}, {30, 0.25f}}), layerCluster({{30, 0.5f}, {10, 1.f}})};
    tracksters_ = {trackster({0}), trackster({1})};
  }

  void candidateWithoutTracksterHasNoHit() { CPPUNIT_ASSERT(truth::recoHits(candidate({}), layerClusters_).empty()); }

  void nullTracksterAddsNoHit() {
    CPPUNIT_ASSERT(truth::recoHits(candidate({edm::Ptr<ticl::Trackster>()}), layerClusters_).empty());
    const auto hits = truth::recoHits(
        candidate({edm::Ptr<ticl::Trackster>(), edm::Ptr<ticl::Trackster>(&tracksters_, 0)}), layerClusters_);
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), hits.size());
  }

  void oneTracksterGivesItsHits() {
    const auto hits = truth::recoHits(candidate({edm::Ptr<ticl::Trackster>(&tracksters_, 1)}), layerClusters_);
    CPPUNIT_ASSERT_EQUAL(std::size_t(2), hits.size());
    CPPUNIT_ASSERT_EQUAL(uint32_t(10), hits[0].detId);
    CPPUNIT_ASSERT_EQUAL(uint32_t(30), hits[1].detId);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.5, hits[1].fraction, 1e-6);
  }

  void sharedCellIsCoalesced() {
    const auto hits = truth::recoHits(
        candidate({edm::Ptr<ticl::Trackster>(&tracksters_, 0), edm::Ptr<ticl::Trackster>(&tracksters_, 1)}),
        layerClusters_);
    CPPUNIT_ASSERT_EQUAL(std::size_t(3), hits.size());
    CPPUNIT_ASSERT_EQUAL(uint32_t(10), hits[0].detId);
    CPPUNIT_ASSERT_EQUAL(uint32_t(20), hits[1].detId);
    CPPUNIT_ASSERT_EQUAL(uint32_t(30), hits[2].detId);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.75, hits[2].fraction, 1e-6);
  }

private:
  std::vector<reco::CaloCluster> layerClusters_;
  std::vector<ticl::Trackster> tracksters_;
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestRecoHitAdapters);
