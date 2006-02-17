// $Id: testRangeMap.cc,v 1.1 2006/02/15 11:57:18 llista Exp $
#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/CopyPolicy.h"


class testRangeMap : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testRangeMap);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testRangeMap);

#include <iostream>
using namespace std;

struct MatchOddId {
  bool operator()( int i ) const { return i % 2 == 1; }
};

void testRangeMap::checkAll() {
  typedef edm::RangeMap<int, std::vector<int>, edm::CopyPolicy<int> > map;

  map m;
  int v3[] = { 1, 2, 3, 4 };
  int v2[] = { 5, 6, 7 };
  int v1[] = { 8, 9 };
  int s1 = sizeof( v1 ) / sizeof( int );
  int s2 = sizeof( v2 ) / sizeof( int );
  int s3 = sizeof( v3 ) / sizeof( int );
  m.put( 2, v2, v2 + s2 );
  m.put( 3, v3,v3+s3 );
  m.put( 1, v1, v1+s1 );

  m.post_insert();
  
  map::const_iterator i = m.begin();
  CPPUNIT_ASSERT( * i ++ == 8 );
  CPPUNIT_ASSERT( * i ++ == 9 );
  CPPUNIT_ASSERT( * i ++ == 5 );
  CPPUNIT_ASSERT( * i ++ == 6 );
  CPPUNIT_ASSERT( * i ++ == 7 );
  CPPUNIT_ASSERT( * i ++ == 1 );
  CPPUNIT_ASSERT( * i ++ == 2 );
  CPPUNIT_ASSERT( * i ++ == 3 );
  CPPUNIT_ASSERT( * i ++ == 4 );
  CPPUNIT_ASSERT( i == m.end() );
  
  CPPUNIT_ASSERT( * -- i == 4 );
  CPPUNIT_ASSERT( * -- i == 3 );
  CPPUNIT_ASSERT( * -- i == 2 );
  CPPUNIT_ASSERT( * -- i == 1 );
  CPPUNIT_ASSERT( * -- i == 7 );
  CPPUNIT_ASSERT( * -- i == 6 );
  CPPUNIT_ASSERT( * -- i == 5 );
  CPPUNIT_ASSERT( * -- i == 9 );
  CPPUNIT_ASSERT( * -- i == 8 );
  CPPUNIT_ASSERT( i == m.begin() );
  
  
  map::id_iterator b = m.id_begin();
  CPPUNIT_ASSERT( m.id_size() == 3 );
  CPPUNIT_ASSERT( * b ++ == 1 );
  CPPUNIT_ASSERT( * b ++ == 2 );
  CPPUNIT_ASSERT( * b ++ == 3 );
  CPPUNIT_ASSERT( b == m.id_end() );

  CPPUNIT_ASSERT( * -- b == 3 );
  CPPUNIT_ASSERT( * -- b == 2 );
  CPPUNIT_ASSERT( * -- b == 1 );
  CPPUNIT_ASSERT( b == m.id_begin() );

  
}
