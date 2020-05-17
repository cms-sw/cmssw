#include <cppunit/extensions/HelperMacros.h>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <iomanip>

#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"

class testPackedGenParticle : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testPackedGenParticle);

  CPPUNIT_TEST(testDefaultConstructor);
  CPPUNIT_TEST(testCopyConstructor);
  CPPUNIT_TEST(testPackUnpack);
  CPPUNIT_TEST(testSimulateReadFromRoot);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}

  void testDefaultConstructor();
  void testCopyConstructor();
  void testPackUnpack();
  void testSimulateReadFromRoot();

private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(testPackedGenParticle);

void testPackedGenParticle::testDefaultConstructor() {
  pat::PackedGenParticle pc;

  CPPUNIT_ASSERT(pc.polarP4() == pat::PackedGenParticle::PolarLorentzVector(0, 0, 0, 0));
  CPPUNIT_ASSERT(pc.p4() == pat::PackedGenParticle::LorentzVector(0, 0, 0, 0));
  CPPUNIT_ASSERT(pc.vertex() == pat::PackedGenParticle::Point(0, 0, 0));
  CPPUNIT_ASSERT(pc.packedPt_ == 0);
  CPPUNIT_ASSERT(pc.packedY_ == 0);
  CPPUNIT_ASSERT(pc.packedPhi_ == 0);
  CPPUNIT_ASSERT(pc.packedM_ == 0);
}

static bool tolerance(double iLHS, double iRHS, double fraction) {
  return std::abs(iLHS - iRHS) <= fraction * std::abs(iLHS + iRHS) / 2.;
}

void testPackedGenParticle::testCopyConstructor() {
  pat::PackedGenParticle::LorentzVector lv(1., 0.5, 0., std::sqrt(1. + 0.25 + 0.120 * 0.120));
  pat::PackedGenParticle::PolarLorentzVector plv(lv.Pt(), lv.Eta(), lv.Phi(), lv.M());

  pat::PackedGenParticle::Point v(0.01, 0.02, 0.);

  pat::PackedGenParticle pc(reco::GenParticle(-1., lv, v, 11, 0, false), edm::Ref<reco::GenParticleCollection>());

  CPPUNIT_ASSERT(tolerance(pc.polarP4().Pt(), plv.Pt(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.polarP4().Eta(), plv.Eta(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.polarP4().Phi(), plv.Phi(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.polarP4().M(), plv.M(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().X(), lv.X(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().Y(), lv.Y(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().Z(), lv.Z(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().E(), lv.E(), 0.001));
  //CPPUNIT_ASSERT(pc.vertex() == v);

  pat::PackedGenParticle copy_pc(pc);

  CPPUNIT_ASSERT(copy_pc.polarP4() == pc.polarP4());
  CPPUNIT_ASSERT(copy_pc.p4() == pc.p4());
  CPPUNIT_ASSERT(copy_pc.vertex() == pc.vertex());
  CPPUNIT_ASSERT(copy_pc.packedPt_ == pc.packedPt_);
  CPPUNIT_ASSERT(copy_pc.packedY_ == pc.packedY_);
  CPPUNIT_ASSERT(copy_pc.packedPhi_ == pc.packedPhi_);
  CPPUNIT_ASSERT(copy_pc.packedM_ == pc.packedM_);

  CPPUNIT_ASSERT(&copy_pc.polarP4() != &pc.polarP4());
  CPPUNIT_ASSERT(&copy_pc.p4() != &pc.p4());
  CPPUNIT_ASSERT(&copy_pc.vertex() != &pc.vertex());
  CPPUNIT_ASSERT(&copy_pc.packedPt_ != &pc.packedPt_);
  CPPUNIT_ASSERT(&copy_pc.packedY_ != &pc.packedY_);
  CPPUNIT_ASSERT(&copy_pc.packedPhi_ != &pc.packedPhi_);
  CPPUNIT_ASSERT(&copy_pc.packedM_ != &pc.packedM_);
}

void testPackedGenParticle::testPackUnpack() {
  pat::PackedGenParticle::LorentzVector lv(1., 1., 0., std::sqrt(2. + 0.120 * 0.120));
  pat::PackedGenParticle::PolarLorentzVector plv(lv.Pt(), lv.Eta(), lv.Phi(), lv.M());

  pat::PackedGenParticle::Point v(-0.005, 0.005, 0.1);

  pat::PackedGenParticle pc(reco::GenParticle(-1., lv, v, 11, 0, false), edm::Ref<reco::GenParticleCollection>());

  CPPUNIT_ASSERT(tolerance(pc.polarP4().Pt(), plv.Pt(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.polarP4().Eta(), plv.Eta(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.polarP4().Phi(), plv.Phi(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.polarP4().M(), plv.M(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().X(), lv.X(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().Y(), lv.Y(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().Z(), lv.Z(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().E(), lv.E(), 0.001));

  pc.pack();

  CPPUNIT_ASSERT(tolerance(pc.polarP4().Pt(), plv.Pt(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.polarP4().Eta(), plv.Eta(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.polarP4().Phi(), plv.Phi(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.polarP4().M(), plv.M(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().X(), lv.X(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().Y(), lv.Y(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().Z(), lv.Z(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().E(), lv.E(), 0.001));
}

void testPackedGenParticle::testSimulateReadFromRoot() {
  pat::PackedGenParticle::LorentzVector lv(1., 1., 0., std::sqrt(2. + 0.120 * 0.120));
  pat::PackedGenParticle::PolarLorentzVector plv(lv.Pt(), lv.Eta(), lv.Phi(), lv.M());

  pat::PackedGenParticle::Point v(-0.005, 0.005, 0.1);

  pat::PackedGenParticle pc(reco::GenParticle(-1., lv, v, 11, 0, false), edm::Ref<reco::GenParticleCollection>());

  CPPUNIT_ASSERT(tolerance(pc.polarP4().Pt(), plv.Pt(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.polarP4().Eta(), plv.Eta(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.polarP4().Phi(), plv.Phi(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.polarP4().M(), plv.M(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().X(), lv.X(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().Y(), lv.Y(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().Z(), lv.Z(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().E(), lv.E(), 0.001));

  //When reading back from ROOT, these were not stored and are nulled out
  delete pc.p4_.exchange(nullptr);
  delete pc.p4c_.exchange(nullptr);

  CPPUNIT_ASSERT(tolerance(pc.polarP4().Pt(), plv.Pt(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.polarP4().Eta(), plv.Eta(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.polarP4().Phi(), plv.Phi(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.polarP4().M(), plv.M(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().X(), lv.X(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().Y(), lv.Y(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().Z(), lv.Z(), 0.001));
  CPPUNIT_ASSERT(tolerance(pc.p4().E(), lv.E(), 0.001));
}
