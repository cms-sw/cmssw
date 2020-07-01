#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/ParameterSet/interface/defaultModuleLabel.h"

class testDefaultModuleLabel : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDefaultModuleLabel);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}

  void test();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDefaultModuleLabel);

void testDefaultModuleLabel::test() {
  CPPUNIT_ASSERT(edm::defaultModuleLabel("Dummy") == "dummy");
  CPPUNIT_ASSERT(edm::defaultModuleLabel("DummyCamelCaps") == "dummyCamelCaps");
  CPPUNIT_ASSERT(edm::defaultModuleLabel("ALLCAPS") == "allcaps");
  CPPUNIT_ASSERT(edm::defaultModuleLabel("STARTCaps") == "startCaps");
  CPPUNIT_ASSERT(edm::defaultModuleLabel("colons::Test") == "colonsTest");
}

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
