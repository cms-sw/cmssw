#include <vector>

#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

class TestRefVector: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestRefVector);
  CPPUNIT_TEST(testIteration);
  CPPUNIT_TEST_SUITE_END();

 public:
  TestRefVector() {}
  ~TestRefVector() {}
  void setUp() {}
  void tearDown() {}

  void testIteration();

 private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestRefVector);

void TestRefVector::testIteration()
{
  typedef std::vector<double> product_t;
  typedef edm::Ref<product_t> ref_t;
  typedef edm::RefVector<product_t> refvec_t;
  
  product_t  product;
  product.push_back(1.0);
  product.push_back(0.5);
  product.push_back(2.0);

  refvec_t  refvec;
  CPPUNIT_ASSERT(refvec.size() == 0);
  CPPUNIT_ASSERT(refvec.empty());

  ref_t    ref0(edm::ProductID(1, 1), &product[0], 0, &product);
  refvec.push_back(ref0);

  ref_t    ref1(edm::ProductID(1, 1), &product[1], 1, &product);
  refvec.push_back(ref1);

  ref_t    ref2(edm::ProductID(1, 1), &product[2], 2, &product);
  refvec.push_back(ref2);

  auto iter = refvec.begin();

  CPPUNIT_ASSERT(iter->id() == edm::ProductID(1,1) && iter->key() == 0 && *(iter->get()) == 1.0);
  ++iter;

  CPPUNIT_ASSERT(iter->id() == edm::ProductID(1,1) && iter->key() == 1 && *(iter->get()) == 0.5);
  ++iter;

  CPPUNIT_ASSERT(iter->id() == edm::ProductID(1,1) && iter->key() == 2 && *(iter->get()) == 2.0);
  ++iter;

  CPPUNIT_ASSERT(iter == refvec.end());
}
