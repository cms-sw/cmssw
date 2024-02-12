/*
 * AOT compilation tests.
 */

#include <stdexcept>
#include <cppunit/extensions/HelperMacros.h>

#include "testBase.h"

class testCompilation : public testBase {
  CPPUNIT_TEST_SUITE(testCompilation);
  CPPUNIT_TEST(test);
  CPPUNIT_TEST_SUITE_END();

public:
  void test() override;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testCompilation);

void testCompilation::test() {
  // create the test model
  std::cout << "creating test model" << std::endl;
  std::string tfaotPath = cmsswPath("/src/PhysicsTools/TensorFlowAOT");
  std::string modelPath = dataPath_ + "/testmodel";
  std::string cmd = "python3 -W ignore";
  cmd += " " + tfaotPath + "/test/create_model.py";
  cmd += " -d " + modelPath;
  runCmd(cmd);

  // compile it
  std::cout << "AOT compile for batch sizes 1, 2" << std::endl;
  std::string compiledModelPath = dataPath_ + "/testmodel_compiled";
  cmd = "PYTHONWARNINGS=ignore cmsml_compile_tf_graph";
  cmd += " " + modelPath;
  cmd += " " + compiledModelPath;
  cmd += " -c 'testmodel_bs{}' 'testmodel_bs{}' ";
  cmd += " -b 1,2";
  runCmd(cmd);

  // check files
  std::cout << "checking output files" << std::endl;
  CPPUNIT_ASSERT(std::filesystem::exists((compiledModelPath + "/aot/testmodel_bs1.h").c_str()));
  CPPUNIT_ASSERT(std::filesystem::exists((compiledModelPath + "/aot/testmodel_bs1.o").c_str()));
  CPPUNIT_ASSERT(std::filesystem::exists((compiledModelPath + "/aot/testmodel_bs2.h").c_str()));
  CPPUNIT_ASSERT(std::filesystem::exists((compiledModelPath + "/aot/testmodel_bs2.o").c_str()));
}
