#include <limits>
#include <string>
#include <vector>

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>

#include "DataFormats/Common/interface/traits.h"

class TestTraits: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestTraits);
  CPPUNIT_TEST(vector_is_happy);
  CPPUNIT_TEST(string_is_happy);
  CPPUNIT_TEST_SUITE_END();

 public:
  TestTraits() {}
  ~TestTraits() {}
  void setUp() {}
  void tearDown() {}

  void vector_is_happy();
  void string_is_happy();

 private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestTraits);


void TestTraits::vector_is_happy()
{
  typedef std::vector<double>::size_type key_type;
  CPPUNIT_ASSERT(edm::key_traits<key_type>::value ==
		 std::numeric_limits<key_type>::max());
  CPPUNIT_ASSERT(edm::key_traits<key_type>::value ==
		 static_cast<key_type>(-1));
}

void TestTraits::string_is_happy()
{
  std::string  const& r = edm::key_traits<std::string>::value;
  CPPUNIT_ASSERT(r.size() == 1);
  CPPUNIT_ASSERT(r[0] == '\a');
}
