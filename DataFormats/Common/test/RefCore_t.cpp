#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>

#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/EDProductGetter.h"

#include "SimpleEDProductGetter.h"

class TestRefCore: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestRefCore);
  CPPUNIT_TEST(default_ctor);

  CPPUNIT_TEST(nondefault_ctor);
  CPPUNIT_TEST_SUITE_END();

 public:
  TestRefCore() { } 
  ~TestRefCore() {}
  void setUp() {}
  void tearDown() {}

  void default_ctor();
  void nondefault_ctor();

 private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestRefCore);

void TestRefCore::default_ctor()
{
  edm::RefCore  default_refcore;
  CPPUNIT_ASSERT(default_refcore.isNull());
  CPPUNIT_ASSERT(default_refcore.isNonnull()==false);
  CPPUNIT_ASSERT(!default_refcore);
  CPPUNIT_ASSERT(default_refcore.productGetter()==0);
  CPPUNIT_ASSERT(default_refcore.id().isValid()==false);
}

void TestRefCore::nondefault_ctor()
{
  SimpleEDProductGetter getter;
  edm::ProductID id(1, 201U);
  CPPUNIT_ASSERT(id.isValid());

  edm::RefCore  refcore(id, 0, &getter, false);
  CPPUNIT_ASSERT(refcore.isNull()==false);
  CPPUNIT_ASSERT(refcore.isNonnull());
  CPPUNIT_ASSERT(!!refcore);
  CPPUNIT_ASSERT(refcore.productGetter()==&getter);
  CPPUNIT_ASSERT(refcore.id().isValid());
}

