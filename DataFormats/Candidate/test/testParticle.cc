// $Id: testParticle.cc,v 1.7 2008/02/21 09:04:10 llista Exp $
#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Candidate/interface/Particle.h"
#include <iostream>

class testParticle : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testParticle);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
  void test( const reco::Particle & t, 
	     reco::Particle::Charge q, const reco::Particle::Point &,
	     const reco::Particle::LorentzVector & p,
	     const reco::Particle::PolarLorentzVector & pp );
};

CPPUNIT_TEST_SUITE_REGISTRATION(testParticle);

void testParticle::test( const reco::Particle & t, 
			 reco::Particle::Charge q, const reco::Particle::Point & vtx,
			 const reco::Particle::LorentzVector & p,
			 const reco::Particle::PolarLorentzVector & pp ) {
  const double epsilon = 1.e-5;
  const double epsilon2 = 5.e-4;
  CPPUNIT_ASSERT( t.charge() == q );
  CPPUNIT_ASSERT( t.threeCharge() == q*3 );
  CPPUNIT_ASSERT( (t.polarP4() - pp).M2() < epsilon );
  CPPUNIT_ASSERT( (t.p4() - p).M2() < epsilon );
  CPPUNIT_ASSERT( fabs(t.p() - p.P()) < epsilon );
  CPPUNIT_ASSERT( fabs(t.energy() - p.E()) < epsilon );
  CPPUNIT_ASSERT( fabs(t.px() - p.Px()) < epsilon );
  CPPUNIT_ASSERT( fabs(t.py() - p.Py()) < epsilon );
  CPPUNIT_ASSERT( fabs(t.pz() - p.Pz()) < epsilon );
  CPPUNIT_ASSERT( fabs(t.phi() - p.Phi()) < epsilon );
  CPPUNIT_ASSERT( fabs(t.theta() - p.Theta()) < epsilon );
  CPPUNIT_ASSERT( fabs(t.eta() - p.Eta()) < epsilon  );
  CPPUNIT_ASSERT( fabs(t.mass() - p.M()) < epsilon );
  CPPUNIT_ASSERT( fabs(t.massSqr() - p.M2()) < epsilon2 );
  CPPUNIT_ASSERT( fabs(t.mt() - p.Mt()) < epsilon );
  CPPUNIT_ASSERT( fabs(t.mtSqr() - p.Mt2()) < epsilon2 );  
  CPPUNIT_ASSERT( fabs(t.rapidity() - p.Rapidity()) < epsilon );  
  CPPUNIT_ASSERT( fabs(t.vx() - vtx.x()) < epsilon );
  CPPUNIT_ASSERT( fabs(t.vy() - vtx.y()) < epsilon );
  CPPUNIT_ASSERT( fabs(t.vz() - vtx.z()) < epsilon );
}

void testParticle::checkAll() {
  const double mass = 123.456;
  const double pz = 100.000;
  const reco::Particle::Point vtx( 6, 7, 8 );
  const reco::Particle::Point vtx1( 8, 7, 6 );
    reco::Particle::Charge q( 1 );
  const reco::Particle::Charge q1( 9 );
  {
    reco::Particle::LorentzVector p( 1.0, 2.0, 3.0, 4.0 );
    reco::Particle::PolarLorentzVector pp(p);
    reco::Particle t( q, p, vtx );
    test(t, q, vtx, p, pp);
    reco::Particle::LorentzVector p1( 1.1, 2.2, 3.3, 4.4 );
    reco::Particle::PolarLorentzVector pp1(p1);
    t.setP4(p1);
    test(t, q, vtx, p1, pp1);
    t.setMass(mass); pp1.SetM(mass); p1 =  reco::Particle::LorentzVector(pp1);
    test(t, q, vtx, p1, pp1);   
    t.setPz(pz); p1.SetPz(pz); pp1 = reco::Particle::PolarLorentzVector(p1);
    test(t, q, vtx, p1, pp1);
    t.setVertex( vtx1 );
    test(t, q, vtx1, p1, pp1);
    t.setCharge( q1 );
    test(t, q1, vtx1, p1, pp1);
    t.setThreeCharge( q1 );
    test(t, q1/3, vtx1, p1, pp1);
  }
  {
    reco::Particle::PolarLorentzVector pp( 1.0, 2.0, 3.0, 4.0 );
    reco::Particle::LorentzVector p(pp);
    reco::Particle t( q, pp, vtx );
    test(t, q, vtx, p, pp);
    reco::Particle::PolarLorentzVector pp1( 1.1, 2.2, 3.3, 4.4 );
    reco::Particle::LorentzVector p1(pp1);
    t.setP4(pp1);
    test(t, q, vtx, p1, pp1);
    t.setMass(mass); pp1.SetM(mass); p1 =  reco::Particle::LorentzVector(pp1);
    test(t, q, vtx, p1, pp1);    
    t.setPz(pz); p1.SetPz(pz); pp1 = reco::Particle::PolarLorentzVector(p1);
    test(t, q, vtx, p1, pp1);   
    t.setVertex( vtx1 );
    test(t, q, vtx1, p1, pp1);
    t.setCharge( q1 );
    test(t, q1, vtx1, p1, pp1);
    t.setThreeCharge( q1 );
    test(t, q1/3, vtx1, p1, pp1);
 }

}
