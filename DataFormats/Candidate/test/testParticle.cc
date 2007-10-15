// $Id: testParticle.cc,v 1.2 2006/05/02 16:13:34 llista Exp $
#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Candidate/interface/Particle.h"

class testParticle : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testParticle);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testParticle);

void testParticle::checkAll() {
  reco::Particle::PolarLorentzVector p( 1.0, 2.0, 3.0, 4.0 );
  reco::Particle::Charge q( 1 );
  reco::Particle t( q, p );
  CPPUNIT_ASSERT( t.charge() == q );
  CPPUNIT_ASSERT( t.polarP4() == p );
  CPPUNIT_ASSERT( t.p4() == reco::Particle::LorentzVector(p) );
  CPPUNIT_ASSERT( t.p() == p.P() );
  CPPUNIT_ASSERT( fabs(t.energy() - p.E()) < 1.e-5 );
  CPPUNIT_ASSERT( fabs(t.px() - p.Px()) < 1.e-5 );
  CPPUNIT_ASSERT( fabs(t.py() - p.Py()) < 1.e-5);
  CPPUNIT_ASSERT( fabs(t.pz() - p.Pz()) < 1.e-5 );
  CPPUNIT_ASSERT( t.phi() == p.Phi() );
  CPPUNIT_ASSERT( fabs(t.theta() == p.Theta()) < 1.e-5 );
  CPPUNIT_ASSERT( t.eta() == p.Eta() );
  const double epsilon = 1.e-9;
  CPPUNIT_ASSERT( fabs( t.mass() - p.M() ) < epsilon );
  CPPUNIT_ASSERT( fabs( t.massSqr() - p.M2() ) < epsilon );
  CPPUNIT_ASSERT( fabs( t.mt() - p.Mt() ) < epsilon );
  CPPUNIT_ASSERT( fabs( t.mtSqr() - p.Mt2() ) < epsilon );
}
