#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "PhysicsTools/MXNet/interface/MXNetPredictor.h"

class testMXNetPredictor : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testMXNetPredictor);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void checkAll();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testMXNetPredictor);

void testMXNetPredictor::checkAll()
{

  // load model and params into BufferFile
  std::string model_path = edm::FileInPath("PhysicsTools/MXNet/test/data/testmxnet-symbol.json").fullPath();
  std::string param_path = edm::FileInPath("PhysicsTools/MXNet/test/data/testmxnet-0000.params").fullPath();

  mxnet::BufferFile *model_file = new mxnet::BufferFile(model_path);
  CPPUNIT_ASSERT(model_file != nullptr);
  CPPUNIT_ASSERT(model_file->GetLength() > 0);

  mxnet::BufferFile *param_file = new mxnet::BufferFile(param_path);
  CPPUNIT_ASSERT(param_file != nullptr);
  CPPUNIT_ASSERT(param_file->GetLength() > 0);

  // create predictor
  mxnet::MXNetPredictor predictor;

  // set input shape
  std::vector<std::string> input_names {"data"};
  std::vector<std::vector<unsigned>> input_shapes {{1, 3}};
  CPPUNIT_ASSERT_NO_THROW( predictor.set_input_shapes(input_names, input_shapes) );

  // load model from BufferFile
  CPPUNIT_ASSERT_NO_THROW( predictor.load_model(model_file, param_file) );

  // run predictor
  std::vector<std::vector<float>> data {{1, 2, 3,}};
  std::vector<float> outputs;
  CPPUNIT_ASSERT_NO_THROW( outputs = predictor.predict(data) );

  // check outputs
  CPPUNIT_ASSERT(outputs.size() == 3);
  CPPUNIT_ASSERT(outputs.at(0) == 42);
  CPPUNIT_ASSERT(outputs.at(1) == 42);
  CPPUNIT_ASSERT(outputs.at(2) == 42);

  delete model_file;
  delete param_file;

}
