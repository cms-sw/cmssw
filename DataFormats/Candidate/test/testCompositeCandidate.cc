#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"

class testCompositeCandidate : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testCompositeCandidate);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testCompositeCandidate);

namespace test {
  class DummyCandidate : public reco::LeafCandidate {
  public:
    DummyCandidate( const LorentzVector & p, Charge q = 0, int x = 0 ) : 
      reco::LeafCandidate(q, p), x_(x) { }
    virtual DummyCandidate * clone() const { return new DummyCandidate( * this ); }
    int x() const { return x_; }
  private:
    int x_;
  };
}

namespace reco {
  GET_DEFAULT_CANDIDATE_COMPONENT( test::DummyCandidate, int, x );
}

namespace test {
  struct DummyCandSelector {
    DummyCandSelector( int x ): x_(x) { }
    bool operator()( const reco::Candidate & c ) const {
      return c.get<int>() == x_;
    }
  private:
    int x_;
  };
}

void testCompositeCandidate::checkAll() {
  reco::Particle::LorentzVector p1( 1.0, 2.0, 3.0, 4.0 ), p2( 1.5, 2.5, 3.5, 4.5 );
  reco::Particle::Charge q1( 1 ), q2( -1 );
  {
    test::DummyCandidate t1( p1, q1 );
    test::DummyCandidate t2( p2, q2 );
    reco::CompositeCandidate c;
    CPPUNIT_ASSERT( c.numberOfDaughters() == 0 );
    c.addDaughter( t1 );
    CPPUNIT_ASSERT( c.numberOfDaughters() == 1 );
    c.addDaughter( t2 );
    CPPUNIT_ASSERT( c.numberOfDaughters() == 2 );
    
    const reco::Candidate * d[ 2 ];
    int idx = 0;
    const reco::CompositeCandidate & cand = c;
    CPPUNIT_ASSERT(c.numberOfDaughters()==2);
    for( reco::Candidate::const_iterator i = cand.begin(); i != cand.end(); ++ i ) {
      d[ idx ++ ] = & * i;
    }
    CPPUNIT_ASSERT( d[ 0 ]->charge() == q1 );
    CPPUNIT_ASSERT( (d[ 0 ]->p4() - p1).M2() < 1.e-4 );
    CPPUNIT_ASSERT( d[ 1 ]->charge() == q2 );
    CPPUNIT_ASSERT( (d[ 1 ]->p4() - p2).M2() < 1.e-4 );
  }
}
