#include <cppunit/extensions/HelperMacros.h>
#include "CommonTools/Utils/interface/MinFunctionSelector.h"
#include "CommonTools/Utils/interface/MaxFunctionSelector.h"
#include "CommonTools/Utils/interface/RangeSelector.h"
#include "CommonTools/Utils/interface/MinObjectPairSelector.h"
#include "CommonTools/Utils/interface/MaxObjectPairSelector.h"
#include "CommonTools/Utils/interface/RangeObjectPairSelector.h"
#include "CommonTools/Utils/interface/PtMinSelector.h"
#include "CommonTools/Utils/interface/EtMinSelector.h"
#include <iostream>

class testSelectors : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testSelectors);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION( testSelectors );

namespace test {
  struct A {
    explicit A( double x ) : x_( x ) { }
    double x() const { return x_; } 
    double pt() const { return x_; }
    double et() const { return x_; }
  private:
    double x_;
  };

  struct Add {
    double operator()( const A & a1, const A& a2 ) const {
      return a1.x() + a2.x();
    }
  };
}

void testSelectors::checkAll() {
  using namespace test;
  {
    A a( 1.0 );
    MinFunctionSelector<A, & A::x> minSel( 0.9 );
    MaxFunctionSelector<A, & A::x> maxSel( 1.1 );
    RangeSelector<A, & A::x> rangeSel( 0.9, 1.1 );
    PtMinSelector ptMinSel( 0.9 );
    EtMinSelector etMinSel( 0.9 );
    
    CPPUNIT_ASSERT( minSel( a ) );
    CPPUNIT_ASSERT( maxSel( a ) );
    CPPUNIT_ASSERT( rangeSel( a ) );
    CPPUNIT_ASSERT( ptMinSel( a ) );
    CPPUNIT_ASSERT( etMinSel( a ) );
  }
  {
    A a1( 3.0 ), a2( 5.0 );
    MinObjectPairSelector<Add> minSel( 7.9 );
    MaxObjectPairSelector<Add> maxSel( 8.1 );
    RangeObjectPairSelector<Add> rangeSel( 7.9, 8.1 );
    CPPUNIT_ASSERT( minSel( a1, a2 ) );
    CPPUNIT_ASSERT( maxSel( a1, a2 ) );
    CPPUNIT_ASSERT( rangeSel( a1, a2 ) );
  }
}
