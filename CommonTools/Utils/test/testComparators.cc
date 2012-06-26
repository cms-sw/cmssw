#include <cppunit/extensions/HelperMacros.h>
#include "CommonTools/Utils/interface/PointerComparator.h"
#include "CommonTools/Utils/interface/PtComparator.h"
#include "CommonTools/Utils/interface/EtComparator.h"
#include <iostream>

class testComparators : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testComparators);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION( testComparators );

namespace test {
  struct A {
    explicit A( double x ) : x_( x ) { }
    double pt() const { return x_; }
    double et() const { return x_; }
    double px() const { return x_; }
    double py() const { return x_; }
    double pz() const { return x_; }
  private:
    double x_;
  };
}

void testComparators::checkAll() {
  using namespace test;
  A a( 1 ), b( 2 );

  LessByPt<A> cmp1;
  LessByEt<A> cmp2;
  NumericSafeLessByPt<A> cmp3;
  NumericSafeLessByEt<A> cmp4;
  CPPUNIT_ASSERT( cmp1( a, b ) ); 
  CPPUNIT_ASSERT( cmp2( a, b ) ); 
  CPPUNIT_ASSERT( cmp3( a, b ) ); 
  CPPUNIT_ASSERT( cmp4( a, b ) ); 

  GreaterByPt<A> cmp5;
  GreaterByEt<A> cmp6;
  NumericSafeGreaterByPt<A> cmp7;
  NumericSafeGreaterByEt<A> cmp8;
  CPPUNIT_ASSERT( cmp5( b, a ) ); 
  CPPUNIT_ASSERT( cmp6( b, a ) ); 
  CPPUNIT_ASSERT( cmp7( b, a ) ); 
  CPPUNIT_ASSERT( cmp8( b, a ) ); 
  PointerComparator<LessByPt<A> > cmp9;
  CPPUNIT_ASSERT( cmp9( &a, &b ) );

}
