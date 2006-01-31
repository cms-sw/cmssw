// $Id: testParticle.cc,v 1.5 2005/12/13 23:27:24 llista Exp $
#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/Candidate/interface/LeafCandidate.h"
#include "PhysicsTools/CandUtils/interface/Thrust.h"
#include <iostream>
using namespace std;
using namespace aod;

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
  CandidateCollection cands;
  cands.push_back( new LeafCandidate( +1, Particle::LorentzVector( 1, 1, 0, 1 ) ) );
  cands.push_back( new LeafCandidate( +1, Particle::LorentzVector( -1, -1, 0, 1 ) ) );
  Thrust t;
  t.compute( cands.begin(), cands.end() );
  cerr << t.thrust() << ", " << t.axis() << endl;
  CPPUNIT_ASSERT( fabs( t.thrust() - 1.0 ) < 1.e-6 );
}
