/*
 * AOT development workflows tests.
 */

#include <stdexcept>
#include <cppunit/extensions/HelperMacros.h>

#include "testBase.h"

class testDevWorkflow : public testBase {
  CPPUNIT_TEST_SUITE(testDevWorkflow);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  void test() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDevWorkflow);

void testDevWorkflow::test() {
  // create the test model
  std::cout << "creating test model" << std::endl;
  std::string tfaotPath = cmsswPath("/src/PhysicsTools/TensorFlowAOT");
  std::string modelPath = dataPath_ + "/testmodel";
  std::string cmd = "python3 -W ignore";
  cmd += " " + tfaotPath + "/test/create_model.py";
  cmd += " -d " + modelPath;
  runCmd(cmd);

  // compile the model with the development workflow
  std::cout << "running development compilation workflow" << std::endl;
  cmd = "python3 -W ignore";
  cmd += " " + tfaotPath + "/scripts/compile_model.py";
  cmd += " -m " + dataPath_ + "/testmodel";
  cmd += " -s PhysicsTools";
  cmd += " -p TensorFlowAOT";
  cmd += " -b 1,2";
  cmd += " -o " + dataPath_ + "/testmodel_compiled";
  runCmd(cmd);

  // check files
  std::cout << "checking output files" << std::endl;
  CPPUNIT_ASSERT(std::filesystem::exists(
      (dataPath_ + "/testmodel_compiled/tfaot-dev-physicstools-tensorflowaot-testmodel.xml").c_str()));
  CPPUNIT_ASSERT(std::filesystem::exists((dataPath_ + "/testmodel_compiled/include/testmodel.h").c_str()));
  CPPUNIT_ASSERT(std::filesystem::exists((dataPath_ + "/testmodel_compiled/include/testmodel_bs1.h").c_str()));
  CPPUNIT_ASSERT(std::filesystem::exists((dataPath_ + "/testmodel_compiled/include/testmodel_bs2.h").c_str()));
  CPPUNIT_ASSERT(std::filesystem::exists((dataPath_ + "/testmodel_compiled/lib/testmodel_bs1.o").c_str()));
  CPPUNIT_ASSERT(std::filesystem::exists((dataPath_ + "/testmodel_compiled/lib/testmodel_bs2.o").c_str()));
}
