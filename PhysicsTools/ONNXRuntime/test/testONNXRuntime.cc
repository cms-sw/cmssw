#include <iostream>

#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#ifdef CMSSW_CUDA_IS_AVAILABLE
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#endif

using namespace cms::Ort;

class testONNXRuntime : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testONNXRuntime);
  CPPUNIT_TEST(checkCPU);
#ifdef CMSSW_CUDA_IS_AVAILABLE
  CPPUNIT_TEST(checkGPU);
#endif
  CPPUNIT_TEST_SUITE_END();

private:
  void test(Backend backend);

public:
  void checkCPU();
  void checkGPU();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testONNXRuntime);

void testONNXRuntime::test(Backend backend) {
  std::string model_path = edm::FileInPath("PhysicsTools/ONNXRuntime/test/data/model.onnx").fullPath();
  auto session_options = ONNXRuntime::defaultSessionOptions(backend);
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

void testONNXRuntime::checkCPU() { test(Backend::cpu); }

// The CUDA execution provider validates the device selected by defaultSessionOptions(), so requesting a device that
// is not available is expected to throw an edm::Exception with the UnavailableAccelerator category.
void testONNXRuntime::checkGPU() {
#ifdef CMSSW_CUDA_IS_AVAILABLE
  if (cms::cudatest::testDevices()) {
    test(Backend::cuda);

    // a device index that is not expected to be available in the job
    constexpr int kUnavailableDevice = 1024;
    bool thrown = false;
    try {
      ONNXRuntime::defaultSessionOptions(Backend::cuda, kUnavailableDevice);
    } catch (edm::Exception const &e) {
      thrown = true;
      CPPUNIT_ASSERT_EQUAL(edm::errors::UnavailableAccelerator, e.categoryCode());
    }
    CPPUNIT_ASSERT(thrown);
  }
#endif
}
