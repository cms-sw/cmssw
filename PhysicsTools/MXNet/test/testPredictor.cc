#include <cppunit/extensions/HelperMacros.h>

#include "PhysicsTools/MXNet/interface/Predictor.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

using namespace mxnet::cpp;

class testMXNetCppPredictor : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testMXNetCppPredictor);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void checkAll();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testMXNetCppPredictor);

void testMXNetCppPredictor::checkAll()
{

  std::string model_path = edm::FileInPath("PhysicsTools/MXNet/test/data/testmxnet-symbol.json").fullPath();
  std::string param_path = edm::FileInPath("PhysicsTools/MXNet/test/data/testmxnet-0000.params").fullPath();

  // load model and params
  Block *block = nullptr;
  CPPUNIT_ASSERT_NO_THROW( block = new Block(model_path, param_path) );
  CPPUNIT_ASSERT(block!=nullptr);

  // create predictor
  Predictor predictor(*block);

  // set input shape
  std::vector<std::string> input_names {"data"};
  std::vector<std::vector<unsigned>> input_shapes {{1, 3}};
  CPPUNIT_ASSERT_NO_THROW( predictor.set_input_shapes(input_names, input_shapes) );

  // run predictor
  std::vector<std::vector<float>> data {{1, 2, 3,}};
  std::vector<float> outputs;
  CPPUNIT_ASSERT_NO_THROW( outputs = predictor.predict(data) );

  // check outputs
  CPPUNIT_ASSERT(outputs.size() == 3);
  CPPUNIT_ASSERT(outputs.at(0) == 42);
  CPPUNIT_ASSERT(outputs.at(1) == 42);
  CPPUNIT_ASSERT(outputs.at(2) == 42);

  delete block;

}
