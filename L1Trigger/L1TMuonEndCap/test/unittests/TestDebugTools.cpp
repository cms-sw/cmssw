#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "L1Trigger/L1TMuonEndCap/interface/DebugTools.h"

class TestDebugTools : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestDebugTools);
  CPPUNIT_TEST(test_assert);
  CPPUNIT_TEST_SUITE_END();

public:
  TestDebugTools() {}
  ~TestDebugTools() override {}
  void setUp() override {}
  void tearDown() override {}

  void test_assert();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestDebugTools);

using namespace emtf;

void TestDebugTools::test_assert() {
  // emtf_assert should not cause an assertion failure in CMSSW production
  CPPUNIT_ASSERT_ASSERTION_PASS(emtf_assert(1 == 2));
}
