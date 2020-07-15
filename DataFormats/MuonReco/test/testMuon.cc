#include <cppunit/extensions/HelperMacros.h>
#include <algorithm>
class testMuon : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testMuon);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override {}
  void tearDown() override {}
  void checkAll();

private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(testMuon);

void testMuon::checkAll() {
  // to be implemented...
}
