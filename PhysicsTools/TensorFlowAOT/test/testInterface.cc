/*
 * AOT interface tests.
 */

#include <stdexcept>
#include <cppunit/extensions/HelperMacros.h>

#include "testBase.h"

#include "PhysicsTools/TensorFlowAOT/interface/AOT.h"

#include "test_models/simplemodel/include/simplemodel.h"
#include "test_models/multimodel/include/multimodel.h"

class testInterface : public testBase {
  CPPUNIT_TEST_SUITE(testInterface);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  void test() override;
  void test_simple();
  void test_multi();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testInterface);

void testInterface::test() {
  test_simple();
  test_multi();
}

void testInterface::test_simple() {
  std::cout << std::endl;
  std::cout << "tesing simplemodel" << std::endl;

  // initialize the model
  auto model = tfaot::Model<PhysicsTools_TensorFlowAOT::simplemodel>();

  // register (optional) batch rules
  model.setBatchRule(1, {1});
  model.setBatchRule(3, {2, 2}, 1);

  // test batching strategies
  CPPUNIT_ASSERT(model.getBatchStrategy().hasRule(1));
  CPPUNIT_ASSERT(model.getBatchStrategy().getRule(1).nSizes() == 1);
  CPPUNIT_ASSERT(model.getBatchStrategy().getRule(1).getLastPadding() == 0);
  CPPUNIT_ASSERT(!model.getBatchStrategy().hasRule(2));
  CPPUNIT_ASSERT(model.getBatchStrategy().hasRule(3));
  CPPUNIT_ASSERT(model.getBatchStrategy().getRule(3).nSizes() == 2);
  CPPUNIT_ASSERT(model.getBatchStrategy().getRule(3).getLastPadding() == 1);
  CPPUNIT_ASSERT(!model.getBatchStrategy().hasRule(4));

  // evaluate batch size 1
  tfaot::FloatArrays input_bs1 = {{0, 1, 2, 3}};
  tfaot::FloatArrays output_bs1;
  std::tie(output_bs1) = model.run<tfaot::FloatArrays>(1, input_bs1);
  CPPUNIT_ASSERT(output_bs1.size() == 1);
  CPPUNIT_ASSERT(output_bs1[0].size() == 2);
  std::cout << std::endl;
  std::cout << "output_bs1[0]: " << output_bs1[0][0] << ", " << output_bs1[0][1] << std::endl;

  // evaluate batch size 2
  tfaot::FloatArrays input_bs2 = {{0, 1, 2, 3}, {4, 5, 6, 7}};
  tfaot::FloatArrays output_bs2;
  std::tie(output_bs2) = model.run<tfaot::FloatArrays>(2, input_bs2);
  CPPUNIT_ASSERT(output_bs2.size() == 2);
  CPPUNIT_ASSERT(output_bs2[0].size() == 2);
  CPPUNIT_ASSERT(output_bs2[1].size() == 2);
  std::cout << "output_bs2[0]: " << output_bs2[0][0] << ", " << output_bs2[0][1] << std::endl;
  std::cout << "output_bs2[1]: " << output_bs2[1][0] << ", " << output_bs2[1][1] << std::endl;

  // there must be a batch rule for size 2 now
  CPPUNIT_ASSERT(model.getBatchStrategy().hasRule(2));

  // evaluate batch size 3
  tfaot::FloatArrays input_bs3 = {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}};
  tfaot::FloatArrays output_bs3;
  std::tie(output_bs3) = model.run<tfaot::FloatArrays>(3, input_bs3);
  CPPUNIT_ASSERT(output_bs3.size() == 3);
  CPPUNIT_ASSERT(output_bs3[0].size() == 2);
  CPPUNIT_ASSERT(output_bs3[1].size() == 2);
  CPPUNIT_ASSERT(output_bs3[2].size() == 2);
  std::cout << "output_bs3[0]: " << output_bs3[0][0] << ", " << output_bs3[0][1] << std::endl;
  std::cout << "output_bs3[1]: " << output_bs3[1][0] << ", " << output_bs3[1][1] << std::endl;
  std::cout << "output_bs3[2]: " << output_bs3[2][0] << ", " << output_bs3[2][1] << std::endl;

  // there still should be no batch rule for size 4
  CPPUNIT_ASSERT(!model.getBatchStrategy().hasRule(4));
}

void testInterface::test_multi() {
  std::cout << std::endl;
  std::cout << "tesing multimodel" << std::endl;

  // initialize the model
  auto model = tfaot::Model<PhysicsTools_TensorFlowAOT::multimodel>();

  // there should be no batch rule for size 2 yet
  CPPUNIT_ASSERT(!model.getBatchStrategy().hasRule(2));

  // evaluate batch size 2
  tfaot::FloatArrays input1_bs2 = {{0, 1, 2, 3}, {4, 5, 6, 7}};
  tfaot::DoubleArrays input2_bs2 = {{0, 1, 2, 3}, {4, 5, 6, 7}};
  tfaot::FloatArrays output1_bs2;
  tfaot::BoolArrays output2_bs2;
  std::tie(output1_bs2, output2_bs2) = model.run<tfaot::FloatArrays, tfaot::BoolArrays>(2, input1_bs2, input2_bs2);
  CPPUNIT_ASSERT(output1_bs2.size() == 2);
  CPPUNIT_ASSERT(output1_bs2[0].size() == 2);
  CPPUNIT_ASSERT(output1_bs2[1].size() == 2);
  std::cout << "output1_bs2[0]: " << output1_bs2[0][0] << ", " << output1_bs2[0][1] << std::endl;
  std::cout << "output1_bs2[1]: " << output1_bs2[1][0] << ", " << output1_bs2[1][1] << std::endl;
  CPPUNIT_ASSERT(output2_bs2.size() == 2);
  CPPUNIT_ASSERT(output2_bs2[0].size() == 2);
  CPPUNIT_ASSERT(output2_bs2[1].size() == 2);
  std::cout << "output2_bs2[0]: " << output2_bs2[0][0] << ", " << output2_bs2[0][1] << std::endl;
  std::cout << "output2_bs2[1]: " << output2_bs2[1][0] << ", " << output2_bs2[1][1] << std::endl;

  // there must be a batch rule for size 2 now
  CPPUNIT_ASSERT(model.getBatchStrategy().hasRule(2));
}
