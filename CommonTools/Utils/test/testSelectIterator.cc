#include <cppunit/extensions/HelperMacros.h>
#include "CommonTools/Utils/interface/PtMinSelector.h"
#include "CommonTools/Utils/interface/Selection.h"
#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>

class testSelectIterator : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testSelectIterator);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION( testSelectIterator );

namespace test {
  struct A {
    explicit A( double x ) : x_( x ) { }
    double pt() const { return x_; }
  private:
    double x_;
  };
}

void testSelectIterator::checkAll() {
  using namespace test;
  using namespace std;
  vector<A> v;
  for( double x = 0; x < 10.1; ++ x )
    v.push_back(A(x));
  CPPUNIT_ASSERT( v.size() == 11 );
  PtMinSelector select( 3.5 );
  Selection<vector<A>, PtMinSelector> sel( v, select );
  CPPUNIT_ASSERT( sel.size() == 7 );
  for( size_t i = 0; i < sel.size(); ++ i ) {
    CPPUNIT_ASSERT( select( sel[i]) );
  }
  for( Selection<vector<A>, PtMinSelector>::const_iterator i = sel.begin(); 
       i != sel.end(); ++ i ) {
    CPPUNIT_ASSERT( select( * i ) );
  }
  vector<A> selected;
  copy( sel.begin(), sel.end(), back_inserter( selected ) );
  CPPUNIT_ASSERT( sel.size() == selected.size() );
  for( size_t i = 0; i < selected.size(); ++ i ) {
    CPPUNIT_ASSERT( select( selected[i]) );
  }
}  

