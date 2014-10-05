// Test of the DictionaryTools functions.


#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"

#include "cppunit/extensions/HelperMacros.h"

#include "Cintex/Cintex.h"

#include <typeinfo>
#include <map>
#include <vector>

class TestDictionaries: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestDictionaries);
  CPPUNIT_TEST(default_is_invalid);
  CPPUNIT_TEST(no_dictionary_is_invalid);
  CPPUNIT_TEST(not_a_template_instance);
  CPPUNIT_TEST(demangling);
  CPPUNIT_TEST_SUITE_END();

 public:
  TestDictionaries() {}
  ~TestDictionaries() {}
  void setUp() {ROOT::Cintex::Cintex::Enable();}
  void tearDown() {}

  void default_is_invalid();
  void no_dictionary_is_invalid();
  void not_a_template_instance();
  void demangling();

 private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestDictionaries);

void TestDictionaries::default_is_invalid() {
  edm::TypeWithDict t;
  CPPUNIT_ASSERT(!t);
}


void TestDictionaries::no_dictionary_is_invalid() {
  edm::TypeWithDict t(edm::TypeWithDict::byName("ThereIsNoTypeWithThisName"));
  CPPUNIT_ASSERT(!t);
}

void TestDictionaries::not_a_template_instance() {
  edm::TypeWithDict not_a_template(edm::TypeWithDict::byName("double"));
  CPPUNIT_ASSERT(not_a_template);
  std::string nonesuch(not_a_template.templateName());
  CPPUNIT_ASSERT(nonesuch.empty());
}

namespace {
  template<typename T>
  void checkIt() {
    edm::TypeWithDict type(typeid(T));
    // Test only if class has dictionary
    if(bool(type)) {
      std::string demangledName(edm::typeDemangle(typeid(T).name()));
      CPPUNIT_ASSERT(type.name() == demangledName);

      edm::TypeID tid(type.typeInfo());
      CPPUNIT_ASSERT(tid.className() == demangledName);

      edm::TypeWithDict typeFromName = edm::TypeWithDict::byName(demangledName);
      edm::TypeID tidFromName(typeFromName.typeInfo());
      CPPUNIT_ASSERT(tidFromName.className() == demangledName);
    }
  }

  template<typename T>
  void checkDemangling() {
    checkIt<std::vector<T> >();
    checkIt<edm::Wrapper<T> >();
    checkIt<edm::Wrapper<std::vector<T> > >();
    checkIt<T>();
  }
}

void TestDictionaries::demangling() {
  checkDemangling<int>();
  checkDemangling<unsigned int>();
  checkDemangling<unsigned long>();
  checkDemangling<long>();
  checkDemangling<unsigned long>();
  checkDemangling<long long>();
  checkDemangling<unsigned long long>();
  checkDemangling<short>();
  checkDemangling<unsigned short>();
  checkDemangling<char>();
  checkDemangling<unsigned char>();
  checkDemangling<float>();
  checkDemangling<double>();
  checkDemangling<bool>();
  checkDemangling<std::string>();
}

