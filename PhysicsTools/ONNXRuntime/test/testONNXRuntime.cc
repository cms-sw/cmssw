#include <cppunit/extensions/HelperMacros.h>

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <chrono>
#include <iostream>

using namespace cms::Ort;

class testONNXRuntime : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testONNXRuntime);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void checkAll();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testONNXRuntime);

void testONNXRuntime::checkAll() {
  std::string model_path = edm::FileInPath("PhysicsTools/ONNXRuntime/test/data/model.onnx").fullPath();
  ONNXRuntime rt(model_path);
  for (const unsigned batch_size : {1, 2, 4}) {
    FloatArrays input_values{
        std::vector<float>(batch_size * 2, 1),
    };
    FloatArrays outputs;
    CPPUNIT_ASSERT_NO_THROW(outputs = rt.run({"X"}, input_values, {}, {"Y"}, batch_size));
    CPPUNIT_ASSERT(outputs.size() == 1);
    CPPUNIT_ASSERT(outputs[0].size() == batch_size);
    for (const auto &v : outputs[0]) {
      CPPUNIT_ASSERT(v == 3);
    }
  }
}
