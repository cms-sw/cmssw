/*
 * TensorFlow AOT test
 * For more info, see https://gitlab.cern.ch/mrieger/CMSSW-DNN.
 *
 * Author: Marcel Rieger
 */

#include <cppunit/extensions/HelperMacros.h>
#include <stdexcept>

#include "testAOT_add/header.h"

using AddComp = testAOT_add;

class testAOT : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testAOT);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void checkAll();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testAOT);

void testAOT::checkAll() {
  {
    std::cout << "testing tf add" << std::endl;
    AddComp add;

    add.arg0() = 1;
    add.arg1() = 2;
    CPPUNIT_ASSERT(add.Run());
    CPPUNIT_ASSERT(add.error_msg() == "");
    CPPUNIT_ASSERT(add.result0() == 3);
    CPPUNIT_ASSERT(add.result0_data()[0] == 3);
    CPPUNIT_ASSERT(add.result0_data() == add.results()[0]);

    add.arg0_data()[0] = 123;
    add.arg1_data()[0] = 456;
    CPPUNIT_ASSERT(add.Run());
    CPPUNIT_ASSERT(add.error_msg() == "");
    CPPUNIT_ASSERT(add.result0() == 579);
    CPPUNIT_ASSERT(add.result0_data()[0] == 579);
    CPPUNIT_ASSERT(add.result0_data() == add.results()[0]);

    const AddComp& add_const = add;
    CPPUNIT_ASSERT(add_const.error_msg() == "");
    CPPUNIT_ASSERT(add_const.arg0() == 123);
    CPPUNIT_ASSERT(add_const.arg0_data()[0] == 123);
    CPPUNIT_ASSERT(add_const.arg1() == 456);
    CPPUNIT_ASSERT(add_const.arg1_data()[0] == 456);
    CPPUNIT_ASSERT(add_const.result0() == 579);
    CPPUNIT_ASSERT(add_const.result0_data()[0] == 579);
    CPPUNIT_ASSERT(add_const.result0_data() == add_const.results()[0]);
  }

  // run tests that use set_argN_data separately, to avoid accidentally re-using
  // non-existent buffers.
  {
    std::cout << "testing tf add no input buffer" << std::endl;
    AddComp add(AddComp::AllocMode::RESULTS_PROFILES_AND_TEMPS_ONLY);

    int arg_x = 10;
    int arg_y = 32;
    add.set_arg0_data(&arg_x);
    add.set_arg1_data(&arg_y);

    CPPUNIT_ASSERT(add.Run());
    CPPUNIT_ASSERT(add.error_msg() == "");
    CPPUNIT_ASSERT(add.result0() == 42);
    CPPUNIT_ASSERT(add.result0_data()[0] == 42);
    CPPUNIT_ASSERT(add.result0_data() == add.results()[0]);
  }
}
