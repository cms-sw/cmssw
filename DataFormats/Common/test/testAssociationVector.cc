// $Id: testAssociationVector.cc,v 1.14 2006/09/26 10:37:22 llista Exp $
#include <cppunit/extensions/HelperMacros.h>
#include <algorithm>
#include <iterator>
#include <iostream>
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/test/TestHandle.h"
using namespace edm;

class testAssociationVector : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testAssociationVector);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
  void dummy();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testAssociationVector);

void testAssociationVector::checkAll() {
  typedef std::vector<double> CKey;
  typedef std::vector<int> CVal;

  CKey k;
  k.push_back( 1.1 );
  k.push_back( 2.2 );
  k.push_back( 3.3 );
  ProductID const pid( 1 );
  TestHandle<CKey> handle( & k, pid );
  RefProd<CKey> ref( handle );
  AssociationVector<CKey, CVal> v( ref );
  v.push_back( 1 );
  v.push_back( 2 );
  v.push_back( 3 );
  CPPUNIT_ASSERT( v.size() == 3 );
  CPPUNIT_ASSERT( v.keyProduct() == ref );
  CPPUNIT_ASSERT( v[ 0 ] == 1 );
  CPPUNIT_ASSERT( v[ 1 ] == 2 );
  CPPUNIT_ASSERT( v[ 2 ] == 3 );
  CPPUNIT_ASSERT( * v.key( 0 ) == 1.1 );
  CPPUNIT_ASSERT( * v.key( 1 ) == 2.2 );
  CPPUNIT_ASSERT( * v.key( 2 ) == 3.3 );
}

// just check that some stuff compiles
void testAssociationVector::dummy() {
  
}
