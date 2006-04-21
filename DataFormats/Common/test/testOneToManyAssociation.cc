// $Id: testOneToManyAssociation.cc,v 1.2 2006/03/28 09:28:39 llista Exp $
#include <cppunit/extensions/HelperMacros.h>
#include <algorithm>
#include <iterator>
#include <iostream>
#include "DataFormats/Common/interface/AssociationMap.h"

class testOneToManyAssociation : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testOneToManyAssociation);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testOneToManyAssociation);

void testOneToManyAssociation::checkAll() {
  typedef std::vector<int> CKey;
  typedef std::vector<double> CVal;
  typedef edm::AssociationMap<CKey, CVal, edm::OneToMany, unsigned char> Assoc;
  Assoc v;
  CPPUNIT_ASSERT( v.empty() );
  CPPUNIT_ASSERT( v.size() == 0 );
  v.insert( edm::Ref<CKey>(), edm::Ref<CVal>() );
  Assoc::const_iterator b = v.begin(), e = v.end();
  CPPUNIT_ASSERT( b != e );
  Assoc::const_iterator f = v.find( edm::Ref<CKey>() );
  CPPUNIT_ASSERT( f != e );
  edm::Ref<Assoc> r;
}
