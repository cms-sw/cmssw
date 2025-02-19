#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
#include <cppunit/extensions/HelperMacros.h>

#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/RefCoreWithIndex.h"
#include "DataFormats/Common/interface/EDProductGetter.h"

#include "SimpleEDProductGetter.h"

class TestRefCore: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestRefCore);
  CPPUNIT_TEST(default_ctor);
  CPPUNIT_TEST(default_ctor_withindex);

  CPPUNIT_TEST(nondefault_ctor);
  CPPUNIT_TEST(nondefault_ctor_withindex);
  CPPUNIT_TEST_SUITE_END();

 public:
  TestRefCore() { } 
  ~TestRefCore() {}
  void setUp() {}
  void tearDown() {}

  void default_ctor();
  void nondefault_ctor();
  void default_ctor_withindex();
  void nondefault_ctor_withindex();

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

void TestRefCore::default_ctor_withindex()
{
  edm::RefCoreWithIndex  default_refcore;
  CPPUNIT_ASSERT(default_refcore.isNull());
  CPPUNIT_ASSERT(default_refcore.isNonnull()==false);
  CPPUNIT_ASSERT(!default_refcore);
  CPPUNIT_ASSERT(default_refcore.productGetter()==0);
  CPPUNIT_ASSERT(default_refcore.id().isValid()==false);
  CPPUNIT_ASSERT(default_refcore.index() == edm::key_traits<unsigned int>::value);
  edm::RefCore compareTo;
  edm::RefCore const& converted = default_refcore.toRefCore();
  CPPUNIT_ASSERT(compareTo.productGetter()==converted.productGetter());
  CPPUNIT_ASSERT(compareTo.id()==converted.id());
}

void TestRefCore::nondefault_ctor_withindex()
{
  SimpleEDProductGetter getter;
  edm::ProductID id(1, 201U);
  CPPUNIT_ASSERT(id.isValid());
  
  edm::RefCoreWithIndex  refcore(id, 0, &getter, false,1);
  CPPUNIT_ASSERT(refcore.isNull()==false);
  CPPUNIT_ASSERT(refcore.isNonnull());
  CPPUNIT_ASSERT(!!refcore);
  CPPUNIT_ASSERT(refcore.productGetter()==&getter);
  CPPUNIT_ASSERT(refcore.id().isValid());
  CPPUNIT_ASSERT(refcore.index() == 1);
  
  edm::RefCore compareTo(id, 0, &getter, false);
  edm::RefCore const& converted = refcore.toRefCore();
  CPPUNIT_ASSERT(compareTo.productGetter()==converted.productGetter());
  CPPUNIT_ASSERT(compareTo.id()==converted.id());
}

