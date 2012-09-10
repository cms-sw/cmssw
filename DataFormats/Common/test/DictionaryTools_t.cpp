// Test of the DictionaryTools functions.


#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"

#include "cppunit/extensions/HelperMacros.h"

#include <iostream>
#include <typeinfo>
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
  CPPUNIT_TEST(primary_template_id);
  CPPUNIT_TEST(not_a_template_instance);
  CPPUNIT_TEST_SUITE_END();

 public:
  TestDictionaries() {}
  ~TestDictionaries() {}
  void setUp() {}
  void tearDown() {}

  void default_is_invalid();
  void no_dictionary_is_invalid();
  void find_nested();
  void burrowing();
  void burrowing_failure();
  void wrapper_type();
  void wrapper_type_failure();
  void primary_template_id();
  void not_a_template_instance();

 private:
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestDictionaries);

void TestDictionaries::default_is_invalid() {
  edm::TypeID t;
  CPPUNIT_ASSERT(!t);
}


void TestDictionaries::no_dictionary_is_invalid() {
  edm::TypeID t(edm::TypeID::byName("ThereIsNoTypeWithThisName"));
  CPPUNIT_ASSERT(!t);
}

void TestDictionaries::find_nested() {
  edm::TypeWithDict intvec(edm::TypeID::byName("std::vector<int>"));
  CPPUNIT_ASSERT(intvec);

  edm::TypeID found_type;

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
  edm::TypeID wrapped_type;
  CPPUNIT_ASSERT(edm::find_nested_type_named("wrapped_type",
                                             wrapper_type,
                                             wrapped_type));
  CPPUNIT_ASSERT(wrapped_type);
  edm::TypeWithDict wrapped_Dict_type(wrapped_type.typeInfo());
  CPPUNIT_ASSERT(!wrapped_Dict_type.isTypedef());
  CPPUNIT_ASSERT(wrapped_Dict_type.isFundamental());
  CPPUNIT_ASSERT(wrapped_type == edm::TypeID::byName("int"));
  CPPUNIT_ASSERT(wrapped_type.typeInfo() == typeid(int));
}

void TestDictionaries::burrowing_failure() {
  edm::TypeWithDict not_a_wrapper(edm::TypeID::byName("double"));
  CPPUNIT_ASSERT(not_a_wrapper);
  edm::TypeID no_such_wrapped_type;
  CPPUNIT_ASSERT(!no_such_wrapped_type);
  CPPUNIT_ASSERT(!edm::find_nested_type_named("wrapped_type",
                                              not_a_wrapper,
                                              no_such_wrapped_type));
  CPPUNIT_ASSERT(!no_such_wrapped_type);
}

void TestDictionaries::wrapper_type() {
  edm::TypeWithDict wrapper_type(typeid(edm::Wrapper<int>));
  edm::TypeID wrapped_type;
  CPPUNIT_ASSERT(edm::wrapper_type_of(wrapper_type, wrapped_type));
  edm::TypeWithDict wrapped_Dict_type(wrapped_type.typeInfo());
  CPPUNIT_ASSERT(!wrapped_Dict_type.isTypedef());
  CPPUNIT_ASSERT(wrapped_type == edm::TypeID::byName("int"));
}

void TestDictionaries::wrapper_type_failure() {
  edm::TypeWithDict not_a_wrapper(edm::TypeID::byName("double"));
  CPPUNIT_ASSERT(not_a_wrapper);
  edm::TypeID no_such_wrapped_type;
  CPPUNIT_ASSERT(!no_such_wrapped_type);
  CPPUNIT_ASSERT(!edm::wrapper_type_of(not_a_wrapper,
                                       no_such_wrapped_type));
  CPPUNIT_ASSERT(!no_such_wrapped_type);
}

void TestDictionaries::primary_template_id() {
  edm::TypeWithDict intvec(edm::TypeWithDict::byName("std::vector<int>"));
  edm::TypeTemplateWithDict vec(intvec);

  // The template std::vector has two template parameters, thus the
  // '2' in the following line.
  edm::TypeTemplateWithDict standard_vec(edm::TypeTemplateWithDict::byName("std::vector",2));
  CPPUNIT_ASSERT(!standard_vec);
  CPPUNIT_ASSERT(!(vec == standard_vec));

  // reflex in use by CMS as of 26 Feb 2007 understands vector to have
  // one template parameter; this is not standard.
  edm::TypeTemplateWithDict nonstandard_vec(edm::TypeTemplateWithDict::byName("std::vector",1));
  CPPUNIT_ASSERT(nonstandard_vec);
  CPPUNIT_ASSERT(vec == nonstandard_vec);
}

void TestDictionaries::not_a_template_instance() {
  edm::TypeWithDict not_a_template(edm::TypeWithDict::byName("double"));
  CPPUNIT_ASSERT(not_a_template);
  edm::TypeTemplateWithDict nonesuch(not_a_template);
  CPPUNIT_ASSERT(!nonesuch);
}

