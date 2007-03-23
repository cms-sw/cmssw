#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "SimpleEDProductGetter.h"

class TestRef: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestRef);
  CPPUNIT_TEST(default_ctor_without_active_getter);
  CPPUNIT_TEST(default_ctor_with_active_getter);
  CPPUNIT_TEST(nondefault_ctor);
  CPPUNIT_TEST_SUITE_END();

 public:
  typedef edm::Ref<std::vector<int> > ref_t;

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
  ref_t  default_refcore;
  CPPUNIT_ASSERT(default_refcore.isNull());
  CPPUNIT_ASSERT(default_refcore.isNonnull()==false);
  CPPUNIT_ASSERT(!default_refcore);
  CPPUNIT_ASSERT(default_refcore.productGetter()==0);
  CPPUNIT_ASSERT(default_refcore.id().isValid()==false);
}

void TestRef::default_ctor_with_active_getter()
{
  SimpleEDProductGetter getter;
  edm::EDProductGetter::Operate op(&getter);
  ref_t  default_refcore;
  CPPUNIT_ASSERT(default_refcore.isNull());
  CPPUNIT_ASSERT(default_refcore.isNonnull()==false);
  CPPUNIT_ASSERT(!default_refcore);
  CPPUNIT_ASSERT(default_refcore.productGetter()==&getter);
  CPPUNIT_ASSERT(default_refcore.id().isValid()==false);
}

void TestRef::nondefault_ctor()
{
  SimpleEDProductGetter getter;
  edm::EDProductGetter::Operate op(&getter);
  edm::ProductID id(201U);
  CPPUNIT_ASSERT(id.isValid());

  ref_t  refcore(id, 0, &getter);
  CPPUNIT_ASSERT(refcore.isNull()==false);
  CPPUNIT_ASSERT(refcore.isNonnull());
  CPPUNIT_ASSERT(!!refcore);
  CPPUNIT_ASSERT(refcore.productGetter()==&getter);
  CPPUNIT_ASSERT(refcore.id().isValid());
}

