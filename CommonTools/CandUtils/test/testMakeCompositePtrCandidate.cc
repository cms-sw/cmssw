// $Id: testMakeCompositePtrCandidate.cc,v 1.2 2009/10/01 09:14:52 llista Exp $
#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "CommonTools/CandUtils/interface/makeCompositeCandidate.h"
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include <iostream>
using namespace reco;
using namespace std;

namespace test {
  class DummyCandidate : public reco::LeafCandidate {
  public:
    DummyCandidate( const LorentzVector & p, Charge q = 0 ) : reco::LeafCandidate( q, p ) { }
    virtual DummyCandidate * clone() const { return new DummyCandidate( * this ); }
  };

  typedef std::vector<DummyCandidate> DummyCandidateCollection;
}

class testMakePtrCompositeCandidate : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testMakePtrCompositeCandidate);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testMakePtrCompositeCandidate);

void testMakePtrCompositeCandidate::checkAll() {
  reco::Particle::LorentzVector p1( 1.0, 2.0, 3.0, 4.0 ), p2( 1.5, 2.5, 3.5, 4.5 );
  reco::Particle::Charge q1( 1 ), q2( -1 );
  test::DummyCandidate t1( p1, q1 );
  test::DummyCandidate t2( p2, q2 );
  edm::ProductID const pid(1, 1);
  test::DummyCandidateCollection dcol;
  dcol.push_back(t1);
  dcol.push_back(t2);
  edm::OrphanHandle<test::DummyCandidateCollection> handle(&dcol, pid);
  reco::CandidatePtr ptr1(handle, 0), ptr2(handle, 1);
  auto_ptr<Candidate> cmp = makeCompositePtrCandidate(ptr1, ptr2)[ AddFourMomenta() ];  
  CPPUNIT_ASSERT(cmp->numberOfDaughters() == 2);
  const reco::Candidate * d[2];
  const Candidate & cmp2 = *cmp;
  d[0] = cmp2.daughter(0);
  d[1] = cmp2.daughter(1);
  CPPUNIT_ASSERT(d[0] != 0);
  CPPUNIT_ASSERT(d[1] != 0);
  const double epsilon = 1.e-5; 
  CPPUNIT_ASSERT( d[0]->charge() == q1 );
  CPPUNIT_ASSERT( fabs(d[0]->p4().pt() - p1.pt()) < epsilon );
  CPPUNIT_ASSERT( fabs(d[0]->p4().eta() - p1.eta()) < epsilon );
  CPPUNIT_ASSERT( fabs(d[0]->p4().phi() - p1.phi()) < epsilon );
  CPPUNIT_ASSERT( d[1]->charge() == q2 );
  CPPUNIT_ASSERT( fabs(d[1]->p4().pt() - p2.pt()) < epsilon );
  CPPUNIT_ASSERT( fabs(d[1]->p4().eta() - p2.eta()) < epsilon );
  CPPUNIT_ASSERT( fabs(d[1]->p4().phi() - p2.phi()) < epsilon );

  reco::Particle::LorentzVector ptot = p1 + p2;
  CPPUNIT_ASSERT( fabs(cmp->pt() - ptot.pt()) < epsilon );
  CPPUNIT_ASSERT( fabs(cmp->eta() - ptot.eta()) < epsilon );
  CPPUNIT_ASSERT( fabs(cmp->phi() - ptot.phi()) < epsilon );

  auto_ptr<Candidate> cmp3 = makeCompositePtrCandidate(ptr1, ptr2, ptr1)[ AddFourMomenta() ];  
  auto_ptr<Candidate> cmp4 = makeCompositePtrCandidate(ptr1, ptr2, ptr1, ptr2)[ AddFourMomenta() ];  
}
