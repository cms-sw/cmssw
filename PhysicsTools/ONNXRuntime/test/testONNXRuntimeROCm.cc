#include <cppunit/extensions/HelperMacros.h>

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "HeterogeneousCore/ROCmUtilities/interface/requireDevices.h"

#include "onnxruntime/onnxruntime_session_options_config_keys.h"

using namespace cms::Ort;

class testONNXRuntimeROCm : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testONNXRuntimeROCm);
  CPPUNIT_TEST(checkROCm);
  CPPUNIT_TEST_SUITE_END();

public:
  void checkROCm();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testONNXRuntimeROCm);

void testONNXRuntimeROCm::checkROCm() {
  if (not cms::rocmtest::testDevices()) {
    return;
  }

  std::string model_path = edm::FileInPath("PhysicsTools/ONNXRuntime/test/data/model.onnx").fullPath();
  auto session_options = ONNXRuntime::defaultSessionOptions(Backend::rocm);
  // make sure that the model runs on the GPU, instead of falling back to the CPU
  session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  ONNXRuntime rt(model_path, &session_options);
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
