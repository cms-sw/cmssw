#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateWithRef.h"
#include <memory>
#include <iostream>

class testCandidate : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testCandidate);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testCandidate);

namespace test {
  
  struct DummyComponent {
    int x;
  };

  struct DummyComponent2 {
    int x;
  };

  class DummyCandidate1 : public reco::LeafCandidate {
  public:
    DummyCandidate1( const LorentzVector & p, Charge q, int x, 
		     int y1, int y2) : reco::LeafCandidate( q, p ) { 
      c.x = x;
      cc[0].x = y1;
      cc[1].x = y2;
    }
    virtual DummyCandidate1 * clone() const { return new DummyCandidate1( * this ); }
    DummyComponent cmp() const { return c; }
    DummyComponent2 cmp2( size_t i ) const { return cc[i]; }
    size_t cmpSize2() const { return 2; }

  private:
    DummyComponent c;
    DummyComponent2 cc[2];
  };

  struct DummyRef {
    bool isNull() const    { return true; }
    bool isNonnull() const { return false; }
    bool operator==( const DummyRef & ref ) const { return true; }
    bool operator!=( const DummyRef & ref ) const { return false; }
  };

}

namespace reco {
  GET_DEFAULT_CANDIDATE_COMPONENT( test::DummyCandidate1, test::DummyComponent, cmp );
  GET_CANDIDATE_MULTIPLECOMPONENTS( test::DummyCandidate1, test::DummyComponent2, cmp2, cmpSize2, DefaultComponentTag );
}

void testCandidate::checkAll() {
  reco::Particle::LorentzVector p( 1.0, 2.0, 3.0, 5.0 );
  GlobalVector v(1.0, 2.0, 3.0);
  reco::LeafCandidate::PolarLorentzVector pl(p);  

  reco::Particle::Charge q( 1 );
  int x = 123, y0 = 111, y1 = 222;
  std::unique_ptr<reco::Candidate> c( new test::DummyCandidate1( p, q, x, y0, y1 ) );
  CPPUNIT_ASSERT( c->charge() == q );
  CPPUNIT_ASSERT( (c->p4() - p).M2() < 1.e-4 );
  CPPUNIT_ASSERT( c->numberOfDaughters() == 0 );
  CPPUNIT_ASSERT( c->get<test::DummyComponent>().x == x );
  CPPUNIT_ASSERT( c->numberOf<test::DummyComponent2>() == 2 );
  CPPUNIT_ASSERT( c->get<test::DummyComponent2>( 0 ).x == y0 );
  CPPUNIT_ASSERT( c->get<test::DummyComponent2>( 1 ).x == y1 );

  reco::CandidateWithRef<test::DummyRef> cwr;
  test::DummyRef dr;
  cwr.setRef( dr );

  reco::Candidate::const_iterator b = c->begin(), e = c->end();
  CPPUNIT_ASSERT( b == e );
  CPPUNIT_ASSERT( e - b == 0 );

  // test constructors

  reco::LeafCandidate c1(q,p);
  reco::LeafCandidate c2(q,pl);
  reco::LeafCandidate c3(q,v,5.f,c1.mass());

  auto ftoi = [](float x)->int { int i; memcpy(&i,&x,4); return i;};
  auto print = [](float a, float b)->bool { std::cout << "\nwhat? " << a <<' ' << b << std::endl; return false;};
  auto ok = [&](float a, float b)->bool { return std::abs(ftoi(a)-ftoi(b))<10 ? true : print(a,b); };

  CPPUNIT_ASSERT(ok(c1.pt(),c2.pt()));
  CPPUNIT_ASSERT(ok(c1.eta(),c2.eta()));
  CPPUNIT_ASSERT(ok(c1.phi(),c2.phi()));
  CPPUNIT_ASSERT(ok(c1.mass(),c2.mass()));

  CPPUNIT_ASSERT(ok(c1.y(),c2.y()));
 
  CPPUNIT_ASSERT(ok(c1.pt(),c3.pt()));
  CPPUNIT_ASSERT(ok(c1.eta(),c3.eta()));
  CPPUNIT_ASSERT(ok(c1.phi(),c3.phi()));
  CPPUNIT_ASSERT(ok(c1.mass(),c3.mass()));

  CPPUNIT_ASSERT(ok(c1.y(),c3.y()));

  CPPUNIT_ASSERT(ok(c1.px(),c2.px()));
  CPPUNIT_ASSERT(ok(c1.py(),c2.py()));
  CPPUNIT_ASSERT(ok(c1.pz(),c2.pz()));
  CPPUNIT_ASSERT(ok(c1.energy(),c2.energy()));

  CPPUNIT_ASSERT(ok(c1.px(),c3.px()));
  CPPUNIT_ASSERT(ok(c1.py(),c3.py()));
  CPPUNIT_ASSERT(ok(c1.pz(),c3.pz()));
  CPPUNIT_ASSERT(ok(c1.energy(),c3.energy()));



}
