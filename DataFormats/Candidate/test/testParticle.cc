// $Id: testParticle.cc,v 1.6 2006/02/21 10:37:33 llista Exp $
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
  reco::Particle::LorentzVector p( 1.0, 2.0, 3.0, 4.0 );
  reco::Particle::Charge q( 1 );
  reco::Particle t( q, p );
  CPPUNIT_ASSERT( t.charge() == q );
  CPPUNIT_ASSERT( t.p4() == p );
  CPPUNIT_ASSERT( t.momentum() == p.P() );
  CPPUNIT_ASSERT( t.energy() == p.E() );
  CPPUNIT_ASSERT( t.px() == p.Px() );
  CPPUNIT_ASSERT( t.py() == p.Py() );
  CPPUNIT_ASSERT( t.pz() == p.Pz() );
  CPPUNIT_ASSERT( t.phi() == p.Phi() );
  CPPUNIT_ASSERT( t.theta() == p.Theta() );
  CPPUNIT_ASSERT( t.eta() == p.Eta() );
  const double epsilon = 1.e-9;
  CPPUNIT_ASSERT( fabs( t.mass() - p.M() ) < epsilon );
  CPPUNIT_ASSERT( fabs( t.massSqr() - p.M2() ) < epsilon );
  CPPUNIT_ASSERT( fabs( t.mt() - p.Mt() ) < epsilon );
  CPPUNIT_ASSERT( fabs( t.mtSqr() - p.Mt2() ) < epsilon );
}
