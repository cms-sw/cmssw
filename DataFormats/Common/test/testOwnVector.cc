// $Id: testOwnVector.cc,v 1.2 2006/02/07 07:01:52 wmtan Exp $
#include <cppunit/extensions/HelperMacros.h>
#include <algorithm>
#include <iterator>
#include <iostream>
#include "DataFormats/Common/interface/OwnVector.h"

class testOwnVector : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testOwnVector);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testOwnVector);

namespace test {
  struct Dummy {
    Dummy( int n, bool * r ) : value( n ), ref( r ) { }
    ~Dummy() { * ref = true; }
    int value;
  private:
    bool * ref;
  };

  struct DummyComp {
    bool operator()( const Dummy& d1, const Dummy& d2 ) {
      return d1.value < d2.value;
    } 
  };

  class a {
  public:
    virtual int f() const = 0;
  };
  class b : public a {
  public:
    b(int i) : ii(i) {;}
    virtual int f() const { return ii;  }
    int ii;
    
  };
  class ss {
  public:
    bool operator() ( const a & a1, const a & a2 ) const { 
      return (a1.f() > a2.f());
    }
  };
  
  std::ostream& operator<<( std::ostream& os, const a & aa ) {
    os << aa.f();
    return os;
  }
}

void testOwnVector::checkAll() {
  edm::OwnVector<test::Dummy> v;
  CPPUNIT_ASSERT( v.size() == 0 );
  CPPUNIT_ASSERT( v.empty() );
  bool deleted[ 3 ] = { false, false, false };
  v.push_back( new test::Dummy( 0, deleted     ) );
  v.push_back( new test::Dummy( 1, deleted + 1 ) );
  v.push_back( new test::Dummy( 2, deleted + 2 ) );
  CPPUNIT_ASSERT( v.size() == 3 );
  edm::OwnVector<test::Dummy>::iterator i;
  i = v.begin();
  edm::OwnVector<test::Dummy>::const_iterator ci = i;
  * ci;
  std::sort( v.begin(), v.end(), test::DummyComp() );
  CPPUNIT_ASSERT( ! v.empty() );
  CPPUNIT_ASSERT( v[ 0 ].value == 0 );
  CPPUNIT_ASSERT( v[ 1 ].value == 1 );
  CPPUNIT_ASSERT( v[ 2 ].value == 2 );
  v.clear();
  CPPUNIT_ASSERT( v.size() == 0 );
  CPPUNIT_ASSERT( v.empty() );
  CPPUNIT_ASSERT( deleted[ 0 ] );
  CPPUNIT_ASSERT( deleted[ 1 ] );
  CPPUNIT_ASSERT( deleted[ 2 ] );
  {
    edm::OwnVector<test::a> v;
    test::a * aa = new test::b(2);
    v.push_back(aa);
    aa = new test::b(1);
    v.push_back(aa);
    aa = new test::b(3);
    v.push_back(aa);
    v.sort( test::ss() );
    std::copy(v.begin(), v.end(), std::ostream_iterator<test::a>(std::cout, "\t"));
  }
}
