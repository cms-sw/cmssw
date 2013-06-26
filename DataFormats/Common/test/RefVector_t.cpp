#include <map>
#include <string>

#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

class TestRefVector: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestRefVector);
  CPPUNIT_TEST(map_string_to_double);
  CPPUNIT_TEST_SUITE_END();

 public:
  TestRefVector() {}
  ~TestRefVector() {}
  void setUp() {}
  void tearDown() {}

  void map_string_to_double();

 private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestRefVector);

void TestRefVector::map_string_to_double()
{
  typedef std::map<std::string, double> product_t;
  typedef edm::Ref<product_t>           ref_t;
  typedef edm::RefVector<product_t>     refvec_t;
  
  product_t  product;
  product["one"]  = 1.0;
  product["half"] = 0.5;
  product["two"]  = 2.0;
  CPPUNIT_ASSERT(product.size() == 3);

  ref_t    ref;

  refvec_t  refvec;
  CPPUNIT_ASSERT(refvec.size() == 0);
  CPPUNIT_ASSERT(refvec.empty());


}
