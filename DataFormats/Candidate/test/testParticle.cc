#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Candidate/interface/Particle.h"
#include "TBufferFile.h"
#include "TClass.h"
#include <iostream>

class testParticle : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testParticle);
  CPPUNIT_TEST(checkXYZ);
  CPPUNIT_TEST(checkPolar);
  CPPUNIT_TEST(checkFloat);
  CPPUNIT_TEST(checkXYZLarge);
  CPPUNIT_TEST(checkPolarLarge);
  CPPUNIT_TEST(checkFloatLarge);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {
  }
  void tearDown() {}
  void checkXYZ();
  void checkPolar();
  void checkFloat();
  void checkXYZLarge();
  void checkPolarLarge();
  void checkFloatLarge();
 
  void testAbs( const reco::Particle & t, 
	     reco::Particle::Charge q, const reco::Particle::Point &,
	     const reco::Particle::LorentzVector & p,
	     const reco::Particle::PolarLorentzVector & pp );
  void testRel( const reco::Particle & t, 
	     reco::Particle::Charge q, const reco::Particle::Point &,
	     const reco::Particle::LorentzVector & p,
	     const reco::Particle::PolarLorentzVector & pp );

  const double mass = 123.456;
  const double pz = 100.000;
  const reco::Particle::Point vtx = reco::Particle::Point( 6, 7, 8 );
  const reco::Particle::Point vtx1 = reco::Particle::Point( 8, 7, 6 );
  const reco::Particle::Charge q= 1;
  const reco::Particle::Charge q1=9;


};

CPPUNIT_TEST_SUITE_REGISTRATION(testParticle);

void testParticle::testAbs( const reco::Particle & t, 
			 reco::Particle::Charge q, const reco::Particle::Point & vtx,
			 const reco::Particle::LorentzVector & p,
			 const reco::Particle::PolarLorentzVector & pp ) {
  const double epsilon = 1.e-5;
  const double epsilon2 = 5.e-4;
  CPPUNIT_ASSERT( t.charge() == q );
  CPPUNIT_ASSERT( t.threeCharge() == q*3 );
  CPPUNIT_ASSERT( (t.polarP4() - pp).M2() < epsilon );
  CPPUNIT_ASSERT( (t.p4() - p).M2() < epsilon );
  CPPUNIT_ASSERT( std::abs(t.p() - p.P()) < epsilon );
  CPPUNIT_ASSERT( std::abs(t.energy() - p.E()) < epsilon );
  CPPUNIT_ASSERT( std::abs(t.px() - p.Px()) < epsilon );
  CPPUNIT_ASSERT( std::abs(t.py() - p.Py()) < epsilon );
  CPPUNIT_ASSERT( std::abs(t.pz() - p.Pz()) < epsilon );
  CPPUNIT_ASSERT( std::abs(t.phi() - p.Phi()) < epsilon );
  CPPUNIT_ASSERT( std::abs(t.theta() - p.Theta()) < epsilon );
  CPPUNIT_ASSERT( std::abs(t.eta() - p.Eta()) < epsilon  );
  CPPUNIT_ASSERT( std::abs(t.mass() - p.M()) < epsilon );
  CPPUNIT_ASSERT( std::abs(t.massSqr() - p.M2()) < epsilon2 );
  CPPUNIT_ASSERT( std::abs(t.mt() - p.Mt()) < epsilon );
  CPPUNIT_ASSERT( std::abs(t.mtSqr() - p.Mt2()) < epsilon2 );  
  CPPUNIT_ASSERT( std::abs(t.rapidity() - p.Rapidity()) < epsilon );  
  CPPUNIT_ASSERT( std::abs(t.vx() - vtx.x()) < epsilon );
  CPPUNIT_ASSERT( std::abs(t.vy() - vtx.y()) < epsilon );
  CPPUNIT_ASSERT( std::abs(t.vz() - vtx.z()) < epsilon );
}

void testParticle::testRel( const reco::Particle & t, 
			 reco::Particle::Charge q, const reco::Particle::Point & vtx,
			 const reco::Particle::LorentzVector & p,
			 const reco::Particle::PolarLorentzVector & pp ) {
  const double epsilon = 1.e-5;
  const double epsilon2 = 5.e-4;
  CPPUNIT_ASSERT( t.charge() == q );
  CPPUNIT_ASSERT( t.threeCharge() == q*3 );
  // these should be zero
  CPPUNIT_ASSERT( (t.polarP4() - pp).M2() < epsilon );
  CPPUNIT_ASSERT( (t.p4() - p).M2() < epsilon );

  CPPUNIT_ASSERT( std::abs(t.p() - p.P())/p.P() < epsilon );
  CPPUNIT_ASSERT( std::abs(t.energy() - p.E())/p.E() < epsilon );

  CPPUNIT_ASSERT( std::abs((t.px() - p.px())/p.px()) < epsilon );
  CPPUNIT_ASSERT( std::abs((t.py() - p.py())/p.py()) < epsilon );
  CPPUNIT_ASSERT( std::abs((t.pz() - p.pz())/p.pz()) < epsilon );
  // here resolution should be absolute no matter what
  CPPUNIT_ASSERT( std::abs(t.phi() - p.Phi()) < epsilon );
  CPPUNIT_ASSERT( std::abs(t.theta() - p.Theta()) < epsilon );
  CPPUNIT_ASSERT( std::abs(t.eta() - p.Eta()) < epsilon  );
  CPPUNIT_ASSERT( std::abs(t.rapidity() - p.Rapidity()) < epsilon );  

  CPPUNIT_ASSERT( std::abs((t.mass() - p.M())/p.M()) < epsilon );
  CPPUNIT_ASSERT( std::abs((t.massSqr() - p.M2())/p.M2()) < epsilon2 );
  CPPUNIT_ASSERT( std::abs((t.mt() - p.Mt())/p.Mt()) < epsilon );
  CPPUNIT_ASSERT( std::abs((t.mtSqr() - p.Mt2())/p.Mt2()) < epsilon2 );  

  // not really tested....  absolute makes sense
  CPPUNIT_ASSERT( std::abs(t.vx() - vtx.x()) < epsilon );
  CPPUNIT_ASSERT( std::abs(t.vy() - vtx.y()) < epsilon );
  CPPUNIT_ASSERT( std::abs(t.vz() - vtx.z()) < epsilon );
}



reco::Particle stream(reco::Particle & t) {
  auto m_class = TClass::GetClass(typeid(reco::Particle));
  TBufferFile wbuffer(TBufferFile::kWrite);
  wbuffer.InitMap();
  wbuffer.StreamObject((&t),m_class);
  
  TBufferFile rbuffer(TBufferFile::kRead, wbuffer.Length(),
		      (wbuffer.Buffer()), kFALSE);
  
  rbuffer.InitMap();

  reco::Particle t2;
  rbuffer.StreamObject(&t2, m_class);
  return t2;
  

}

void testParticle::checkXYZ() {
  reco::Particle::LorentzVector p( 1.0, 2.0, 3.0, 4.0 );
  reco::Particle::PolarLorentzVector pp(p);
  reco::Particle t( q, p, vtx );
  testAbs(t, q, vtx, p, pp);
  
  auto t2 = stream(t);
  testAbs(t2, q, vtx, p, pp);
 

  reco::Particle::LorentzVector p1( 1.1, 2.2, 3.3, 4.4 );
  reco::Particle::PolarLorentzVector pp1(p1);
  t.setP4(p1);
  testAbs(t, q, vtx, p1, pp1);
  t.setMass(mass); pp1.SetM(mass); p1 =  reco::Particle::LorentzVector(pp1);
  testAbs(t, q, vtx, p1, pp1);   
  t.setPz(pz); p1.SetPz(pz); pp1 = reco::Particle::PolarLorentzVector(p1);
  testAbs(t, q, vtx, p1, pp1);
  t.setVertex( vtx1 );
  testAbs(t, q, vtx1, p1, pp1);
  t.setCharge( q1 );
  testAbs(t, q1, vtx1, p1, pp1);
  t.setThreeCharge( q1 );
  testAbs(t, q1/3, vtx1, p1, pp1);

 
}

  void testParticle::checkPolar(){
    reco::Particle::PolarLorentzVector pp( 1.0, 2.0, 3.0, 4.0 );
    reco::Particle::LorentzVector p(pp);
    reco::Particle t( q, pp, vtx );
    testAbs(t, q, vtx, p, pp);

    auto t2 = stream(t);
    testAbs(t2, q, vtx, p, pp);


    reco::Particle::PolarLorentzVector pp1( 1.1, 2.2, 3.3, 4.4 );
    reco::Particle::LorentzVector p1(pp1);
    t.setP4(pp1);
    testAbs(t, q, vtx, p1, pp1);
    t.setMass(mass); pp1.SetM(mass); p1 =  reco::Particle::LorentzVector(pp1);
    testAbs(t, q, vtx, p1, pp1);    
    t.setPz(pz); p1.SetPz(pz); pp1 = reco::Particle::PolarLorentzVector(p1);
    testAbs(t, q, vtx, p1, pp1);   
    t.setVertex( vtx1 );
    testAbs(t, q, vtx1, p1, pp1);
    t.setCharge( q1 );
    testAbs(t, q1, vtx1, p1, pp1);
    t.setThreeCharge( q1 );
    testAbs(t, q1/3, vtx1, p1, pp1);
 }

  void testParticle::checkFloat(){
    reco::Particle::PolarLorentzVector pp( 1.0, 2.0, 3.0, 4.0 );
    PtEtaPhiMass pepm(1.0f, 2.0f, 3.0f, 4.0f );
    reco::Particle::LorentzVector p(pp);
    reco::Particle t( q, pepm, vtx );
    
    testAbs(t, q, vtx, p, pp);    

    auto t2 = stream(t);
    testAbs(t2, q, vtx, p, pp);


    reco::Particle::PolarLorentzVector pp1( 1.1, 2.2, 3.3, 4.4 );
    reco::Particle::LorentzVector p1(pp1);
    t.setP4(pp1);
    testAbs(t, q, vtx, p1, pp1);
    t.setMass(mass); pp1.SetM(mass); p1 =  reco::Particle::LorentzVector(pp1);
    testAbs(t, q, vtx, p1, pp1);    
    t.setPz(pz); p1.SetPz(pz); pp1 = reco::Particle::PolarLorentzVector(p1);
    testAbs(t, q, vtx, p1, pp1);   
    t.setVertex( vtx1 );
    testAbs(t, q, vtx1, p1, pp1);
    t.setCharge( q1 );
    testAbs(t, q1, vtx1, p1, pp1);
    t.setThreeCharge( q1 );
    testAbs(t, q1/3, vtx1, p1, pp1);
   
 }

  void testParticle::checkXYZLarge() {
    auto E = std::sqrt(321.*321. + 163.*163. + 678*678. + 11.21*11.21);
    reco::Particle::LorentzVector p( 321., 163., 678.,E );
    reco::Particle::PolarLorentzVector pp(p);
    reco::Particle t( q, p, vtx );
    testRel(t, q, vtx, p, pp);

    auto t2 = stream(t);
    testRel(t2, q, vtx, p, pp);


    reco::Particle::LorentzVector p1 (321.1, 163.1, 678.1, E+0.5 );
    reco::Particle::PolarLorentzVector pp1(p1);
    t.setP4(p1);
    testRel(t, q, vtx, p1, pp1);
    t.setMass(mass); pp1.SetM(mass); p1 =  reco::Particle::LorentzVector(pp1);
    testRel(t, q, vtx, p1, pp1);   
    t.setPz(pz); p1.SetPz(pz); pp1 = reco::Particle::PolarLorentzVector(p1);
    testRel(t, q, vtx, p1, pp1);
    t.setVertex( vtx1 );
    testRel(t, q, vtx1, p1, pp1);
    t.setCharge( q1 );
    testRel(t, q1, vtx1, p1, pp1);
    t.setThreeCharge( q1 );
    testRel(t, q1/3, vtx1, p1, pp1);




  }

  void testParticle::checkPolarLarge(){
    reco::Particle::PolarLorentzVector pp( 1234.56, 1.96, 2.45, 11.21 );
    reco::Particle::LorentzVector p(pp);
    reco::Particle t( q, pp, vtx );
    testRel(t, q, vtx, p, pp);

    auto t2 = stream(t);
    testRel(t2, q, vtx, p, pp);


    reco::Particle::PolarLorentzVector pp1( 1234.76, 1.86, 2.75, 11.11 );
    reco::Particle::LorentzVector p1(pp1);
    t.setP4(pp1);
    testRel(t, q, vtx, p1, pp1);
    t.setMass(mass); pp1.SetM(mass); p1 =  reco::Particle::LorentzVector(pp1);
    testRel(t, q, vtx, p1, pp1);    
    t.setPz(pz); p1.SetPz(pz); pp1 = reco::Particle::PolarLorentzVector(p1);
    testRel(t, q, vtx, p1, pp1);   
    t.setVertex( vtx1 );
    testRel(t, q, vtx1, p1, pp1);
    t.setCharge( q1 );
    testRel(t, q1, vtx1, p1, pp1);
    t.setThreeCharge( q1 );
    testRel(t, q1/3, vtx1, p1, pp1);
 }

  void testParticle::checkFloatLarge(){
    reco::Particle::PolarLorentzVector pp( 1234.56, 1.96, 2.45, 11.21 );
    PtEtaPhiMass pepm(1234.56f, 1.96f, 2.45f, 11.21f );
    reco::Particle::LorentzVector p(pp);
    reco::Particle t( q, pepm, vtx );
    
    testRel(t, q, vtx, p, pp);

    auto t2 = stream(t);
    testRel(t2, q, vtx, p, pp);

    
    reco::Particle::PolarLorentzVector pp1( 1.1, 2.2, 3.3, 4.4 );
    reco::Particle::LorentzVector p1(pp1);
    t.setP4(pp1);
    testRel(t, q, vtx, p1, pp1);
    t.setMass(mass); pp1.SetM(mass); p1 =  reco::Particle::LorentzVector(pp1);
    testRel(t, q, vtx, p1, pp1);    
    t.setPz(pz); p1.SetPz(pz); pp1 = reco::Particle::PolarLorentzVector(p1);
    testRel(t, q, vtx, p1, pp1);   
    t.setVertex( vtx1 );
    testRel(t, q, vtx1, p1, pp1);
    t.setCharge( q1 );
    testRel(t, q1, vtx1, p1, pp1);
    t.setThreeCharge( q1 );
    testRel(t, q1/3, vtx1, p1, pp1);
    
 }

