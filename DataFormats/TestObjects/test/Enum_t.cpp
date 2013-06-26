// Test of the DictionaryTools functions.


#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"

#include "cppunit/extensions/HelperMacros.h"

#include <typeinfo>
#include <vector>

class TestDictionaries: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestDictionaries);
  CPPUNIT_TEST(enum_is_valid);
  CPPUNIT_TEST(enum_by_name_is_valid);
  CPPUNIT_TEST(demangling);
  CPPUNIT_TEST_SUITE_END();

 public:
  TestDictionaries() {}
  ~TestDictionaries() {}
  void setUp() {}
  void tearDown() {}

  void enum_is_valid();
  void enum_by_name_is_valid();
  void demangling();

 private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestDictionaries);

void TestDictionaries::enum_is_valid() {
  edm::TypeWithDict t(typeid(edmtest::EnumProduct::TheEnumProduct));
  CPPUNIT_ASSERT(t);
}

void TestDictionaries::enum_by_name_is_valid() {
  edm::TypeWithDict t = edm::TypeWithDict::byName("edmtest::EnumProduct::TheEnumProduct");
  CPPUNIT_ASSERT(t);
}

namespace {
  template<typename T>
  void checkIt() {
    edm::TypeWithDict type(typeid(T));
    // Test only if class has dictionary
    if(bool(type)) {
      std::string demangledName(edm::typeDemangle(typeid(T).name()));
      CPPUNIT_ASSERT(type.name() == demangledName);
    }
  }

  template<typename T>
  void checkDemangling() {
    checkIt<T>();
    checkIt<std::vector<T> >();
  }
}

void TestDictionaries::demangling() {
  checkDemangling<edmtest::EnumProduct::TheEnumProduct>();
}

