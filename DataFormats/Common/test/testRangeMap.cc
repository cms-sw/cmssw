// $Id: testIDVectorMap.cc,v 1.4 2006/02/14 11:16:04 llista Exp $
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

void testRangeMap::checkAll() {
  typedef edm::RangeMap<int, std::vector<int>, edm::CopyPolicy<int> > map;
  // no test implemented yet...
  
}
