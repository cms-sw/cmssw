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
  CPPUNIT_TEST(find_nested);
  CPPUNIT_TEST(burrowing);
  CPPUNIT_TEST(burrowing_failure);
  CPPUNIT_TEST(wrapper_type);
  CPPUNIT_TEST(wrapper_type_failure);
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
  void find_nested();
  void burrowing();
  void burrowing_failure();
  void wrapper_type();
  void wrapper_type_failure();
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

void TestDictionaries::find_nested() {
  edm::TypeWithDict intvec(edm::TypeWithDict::byName("std::vector<int>"));
  CPPUNIT_ASSERT(intvec);

  edm::TypeWithDict found_type;

  CPPUNIT_ASSERT(edm::find_nested_type_named("const_iterator",
                                             intvec,
                                             found_type));

  CPPUNIT_ASSERT(!edm::find_nested_type_named("WankelRotaryEngine",
                                              intvec,
                                              found_type));
}

void TestDictionaries::burrowing() {
  edm::TypeWithDict wrapper_type(typeid(edm::Wrapper<int>));
  CPPUNIT_ASSERT(wrapper_type);
  edm::TypeWithDict wrapped_type;
  CPPUNIT_ASSERT(edm::find_nested_type_named("wrapped_type",
                                             wrapper_type,
                                             wrapped_type));
  CPPUNIT_ASSERT(wrapped_type);
  edm::TypeWithDict wrapped_Dict_type(wrapped_type.typeInfo());
  CPPUNIT_ASSERT(!wrapped_Dict_type.isTypedef());
  CPPUNIT_ASSERT(wrapped_Dict_type.isFundamental());
  CPPUNIT_ASSERT(wrapped_type == edm::TypeWithDict::byName("int"));
  CPPUNIT_ASSERT(wrapped_type.typeInfo() == typeid(int));
}

void TestDictionaries::burrowing_failure() {
  edm::TypeWithDict not_a_wrapper(edm::TypeWithDict::byName("double"));
  CPPUNIT_ASSERT(not_a_wrapper);
  edm::TypeWithDict no_such_wrapped_type;
  CPPUNIT_ASSERT(!no_such_wrapped_type);
  CPPUNIT_ASSERT(!edm::find_nested_type_named("wrapped_type",
                                              not_a_wrapper,
                                              no_such_wrapped_type));
  CPPUNIT_ASSERT(!no_such_wrapped_type);
}

void TestDictionaries::wrapper_type() {
  edm::TypeWithDict wrapper_type(typeid(edm::Wrapper<int>));
  edm::TypeWithDict wrapped_type;
  CPPUNIT_ASSERT(edm::wrapper_type_of(wrapper_type, wrapped_type));
  edm::TypeWithDict wrapped_Dict_type(wrapped_type.typeInfo());
  CPPUNIT_ASSERT(!wrapped_Dict_type.isTypedef());
  CPPUNIT_ASSERT(wrapped_type == edm::TypeWithDict::byName("int"));
}

void TestDictionaries::wrapper_type_failure() {
  edm::TypeWithDict not_a_wrapper(edm::TypeWithDict::byName("double"));
  CPPUNIT_ASSERT(not_a_wrapper);
  edm::TypeWithDict no_such_wrapped_type;
  CPPUNIT_ASSERT(!no_such_wrapped_type);
  CPPUNIT_ASSERT(!edm::wrapper_type_of(not_a_wrapper,
                                       no_such_wrapped_type));
  CPPUNIT_ASSERT(!no_such_wrapped_type);
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

