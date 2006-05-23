// $Id: testOneToOneAssociation.cc,v 1.2 2006/03/28 09:28:39 llista Exp $
#include <cppunit/extensions/HelperMacros.h>
#include <algorithm>
#include <iterator>
#include <iostream>
#include "DataFormats/Common/interface/OneToOneAssociation.h"
#include "DataFormats/Common/interface/Ref.h"

class testOneToOneAssociation : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testOneToOneAssociation);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testOneToOneAssociation);

void testOneToOneAssociation::checkAll() {
  typedef std::vector<int> CKey;
  typedef std::vector<double> CVal;
  typedef edm::OneToOneAssociation<CKey, CVal, unsigned char> Assoc;
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
