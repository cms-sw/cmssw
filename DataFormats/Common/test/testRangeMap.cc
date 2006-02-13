// $Id: testMapRange.cc,v 1.1 2006/02/01 18:15:01 llista Exp $
#include <cppunit/extensions/HelperMacros.h>
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/CopyPolicy.h"

class testMapRange : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testMapRange);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testMapRange);

#include <iostream>
using namespace std;

void testMapRange::checkAll() {
  typedef edm::RangeMap<int, std::vector<int>, edm::CopyPolicy<int> > map;
  map m;
  int v1[] = { 1, 2, 3, 4 };
  int v2[] = { 5, 6, 7 };
  int v3[] = { 8, 9 };
  int s1 = sizeof( v1 ) / sizeof( int );
  int s2 = sizeof( v2 ) / sizeof( int );
  int s3 = sizeof( v3 ) / sizeof( int );
  m.put( 1, v1, v1 + s1 );
  m.put( 2, v2, v2 + s2 );
  m.put( 3, v3, v3 + s3 );
  map::range r1 = m.get( 1 );
  map::range r2 = m.get( 2 );
  map::range r3 = m.get( 3 );
  map::const_iterator i;

  CPPUNIT_ASSERT( r1.second - r1.first == s1 );
  i = r1.first;
  CPPUNIT_ASSERT( *i++ == v1[ 0 ] );
  CPPUNIT_ASSERT( *i++ == v1[ 1 ] );
  CPPUNIT_ASSERT( *i++ == v1[ 2 ] );
  CPPUNIT_ASSERT( *i++ == v1[ 3 ] );
  CPPUNIT_ASSERT( i == r1.second );

  CPPUNIT_ASSERT( r2.second - r2.first == s2 );
  i = r2.first;
  CPPUNIT_ASSERT( *i++ == v2[ 0 ] );
  CPPUNIT_ASSERT( *i++ == v2[ 1 ] );
  CPPUNIT_ASSERT( *i++ == v2[ 2 ] );
  CPPUNIT_ASSERT( i == r2.second );

  CPPUNIT_ASSERT( r3.second - r3.first == s3 );
  i = r3.first;
  CPPUNIT_ASSERT( *i++ == v3[ 0 ] );
  CPPUNIT_ASSERT( *i++ == v3[ 1 ] );
  CPPUNIT_ASSERT( i == r3.second );

  CPPUNIT_ASSERT( m.size() == size_t( s1 + s2 + s3 ) );

  map::id_iterator b = m.id_begin();
  map::id_iterator e = m.id_end();
  CPPUNIT_ASSERT( m.id_size() == 3 );
  CPPUNIT_ASSERT( * b == 1 );
  ++b;
  CPPUNIT_ASSERT( *b == 2 );
  ++b;
  CPPUNIT_ASSERT( *b == 3 );
  ++b;
  CPPUNIT_ASSERT( b == e );
  
}
