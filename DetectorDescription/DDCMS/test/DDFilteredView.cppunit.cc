#include <cppunit/extensions/HelperMacros.h>

#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"

#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

class testDDFilteredView : public CppUnit::TestFixture {
  
  CPPUNIT_TEST_SUITE(testDDFilteredView);
  CPPUNIT_TEST(checkFilteredView);
  CPPUNIT_TEST_SUITE_END();

public:

  void setUp() override {}
  void tearDown() override {}
  void checkFilteredView();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDDFilteredView);

void testDDFilteredView::checkFilteredView()
{
  cms::DDFilteredView fview();
}
