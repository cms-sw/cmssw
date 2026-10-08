#include <cppunit/extensions/HelperMacros.h>

#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "testONNXRuntime.h"

using namespace cms::Ort;

class testONNXRuntimeCUDA : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testONNXRuntimeCUDA);
  CPPUNIT_TEST(checkCUDA);
  CPPUNIT_TEST_SUITE_END();

public:
  void checkCUDA();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testONNXRuntimeCUDA);

// The CUDA execution provider validates the device selected by defaultSessionOptions(), so requesting a device that
// is not available is expected to throw.
void testONNXRuntimeCUDA::checkCUDA() {
  testGPUBackend(Backend::cuda, "CUDAService", cms::cudatest::testDevices, true);
}
