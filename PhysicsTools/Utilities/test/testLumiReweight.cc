#include <cppunit/extensions/HelperMacros.h>
#include "PhysicsTools/Utilities/interface/LumiReweightingStandAlone.h"

class testLumiReweight : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testLumiReweight);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}
  void checkAll();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testLumiReweight);

void testLumiReweight::checkAll() {}
