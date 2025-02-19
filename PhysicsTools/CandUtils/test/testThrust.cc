// $Id: testThrust.cc,v 1.8 2008/03/13 13:28:00 llista Exp $
#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "PhysicsTools/CandUtils/interface/Thrust.h"
using namespace reco;

class testThrust : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testThrust);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testThrust);

void testThrust::checkAll() {
  {
    std::vector<LeafCandidate> cands;
    cands.push_back(LeafCandidate(+1, Particle::LorentzVector(1, 1, 0, 1)));
    cands.push_back(LeafCandidate(+1, Particle::LorentzVector(-1, -1, 0, 1)));
    Thrust t(cands.begin(), cands.end());
    CPPUNIT_ASSERT(fabs(t.thrust() - 1.0) < 1.e-6);
  }
  {
    std::vector<LeafCandidate> cands;
    cands.push_back(LeafCandidate(+1, Particle::LorentzVector(0.7, 0.5, 0, 1)));
    cands.push_back(LeafCandidate(+1, Particle::LorentzVector(-0.7, -0.5, 0, 1)));
    Thrust t(cands.begin(), cands.end());
    CPPUNIT_ASSERT(fabs(t.thrust() - 1.0) < 1.e-6);
  }
  {
    std::vector<LeafCandidate> cands;
    cands.push_back(LeafCandidate(+1, Particle::LorentzVector(1, 0, 0, 1)));
    cands.push_back(LeafCandidate(+1, Particle::LorentzVector(0, 1, 0, 1)));
    Thrust t(cands.begin(), cands.end());
    CPPUNIT_ASSERT(fabs(t.thrust() - sqrt(2.0)/2) < 1.e-6);
  }
  {
    std::vector<LeafCandidate> cands;
    cands.push_back(LeafCandidate(+1, Particle::LorentzVector(1, 0, 0, 1)));
    cands.push_back(LeafCandidate(+1, Particle::LorentzVector(0, 1, 0, 1)));
    cands.push_back(LeafCandidate(+1, Particle::LorentzVector(1, 1, 0, 1)));
    Thrust t(cands.begin(), cands.end());
    CPPUNIT_ASSERT(t.thrust() > 0.5);
    CPPUNIT_ASSERT(fabs(t.axis().z()) < 1.e-4);
  }
}
