#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "SimpleEDProductGetter.h"

#include <iostream>

class TestRef: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestRef);
  CPPUNIT_TEST(default_ctor_without_active_getter);
  CPPUNIT_TEST(default_ctor_with_active_getter);
  CPPUNIT_TEST(nondefault_ctor);
  CPPUNIT_TEST_SUITE_END();

 public:
  typedef std::vector<int>        product_t;
  typedef edm::Wrapper<product_t> wrapper_t;
  typedef edm::Ref<product_t>     ref_t;

  TestRef() { } 
  ~TestRef() {}
  void setUp() {}
  void tearDown() {}

  void default_ctor_without_active_getter();
  void default_ctor_with_active_getter();
  void nondefault_ctor();

 private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestRef);

void TestRef::default_ctor_without_active_getter()
{
  ref_t  default_ref;
  CPPUNIT_ASSERT(default_ref.isNull());
  CPPUNIT_ASSERT(default_ref.isNonnull()==false);
  CPPUNIT_ASSERT(!default_ref);
  CPPUNIT_ASSERT(default_ref.productGetter()==0);
  CPPUNIT_ASSERT(default_ref.id().isValid()==false);
}

void TestRef::default_ctor_with_active_getter()
{
  SimpleEDProductGetter getter;
  edm::EDProductGetter::Operate op(&getter);
  ref_t  default_ref;
  CPPUNIT_ASSERT(default_ref.isNull());
  CPPUNIT_ASSERT(default_ref.isNonnull()==false);
  CPPUNIT_ASSERT(!default_ref);
  CPPUNIT_ASSERT(default_ref.productGetter()==&getter);
  CPPUNIT_ASSERT(default_ref.id().isValid()==false);
  CPPUNIT_ASSERT_THROW(default_ref.operator->(), edm::Exception);
  CPPUNIT_ASSERT_THROW(*default_ref, edm::Exception);
}

void TestRef::nondefault_ctor()
{
  SimpleEDProductGetter getter;
  
  edm::EDProductGetter::Operate op(&getter);
  edm::ProductID id(201U);
  CPPUNIT_ASSERT(id.isValid());

  std::auto_ptr<product_t> prod(new product_t);
  prod->push_back(1);
  prod->push_back(2);
  getter.addProduct(id, prod);


  ref_t  ref0(id, 0, &getter);
  CPPUNIT_ASSERT(ref0.isNull()==false);
  CPPUNIT_ASSERT(ref0.isNonnull());
  CPPUNIT_ASSERT(!!ref0);
  CPPUNIT_ASSERT(ref0.productGetter()==&getter);
  CPPUNIT_ASSERT(ref0.id().isValid());
  CPPUNIT_ASSERT(*ref0 == 1);

  ref_t  ref1(id, 1, &getter);
  CPPUNIT_ASSERT(ref1.isNonnull());
  CPPUNIT_ASSERT(*ref1 == 2);

  // Note that nothing stops one from making an edm::Ref into a
  // collection using an index that is invalid. So there is no testing
  // of such use to be done.
}

