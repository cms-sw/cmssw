// $Id: testValueMap.cc,v 1.6 2006/06/14 10:46:44 llista Exp $
#include <cppunit/extensions/HelperMacros.h>
#include <algorithm>
#include <iterator>
#include <iostream>
#include "DataFormats/Common/interface/AssociationMap.h"

class testValueMap : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testValueMap);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
  void dummy();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testValueMap);

void testValueMap::checkAll() {
  typedef std::vector<int> CKey;
  typedef double Val;
  typedef edm::AssociationMap<edm::OneToValue<CKey, Val, unsigned char> > Assoc;
  Assoc v;
  CPPUNIT_ASSERT( v.empty() );
  CPPUNIT_ASSERT( v.size() == 0 );
}

// just check that some stuff compiles
void  testValueMap::dummy() {
  typedef std::vector<int> CKey;
  typedef double Val;
  typedef edm::AssociationMap<edm::OneToValue<CKey, Val, unsigned char> > Assoc;
  Assoc v;
  v.insert( edm::Ref<CKey>(), 3.145 );
  Assoc::const_iterator b = v.begin(), e = v.end();
  b++; e++;
  Assoc::const_iterator f = v.find( edm::Ref<CKey>() );
  v.numberOfAssociations( edm::Ref<CKey>() );
  f++;
  edm::Ref<Assoc> r;
  v[ edm::Ref<CKey>() ];
  v.erase( edm::Ref<CKey>() );
  v.clear();
}
