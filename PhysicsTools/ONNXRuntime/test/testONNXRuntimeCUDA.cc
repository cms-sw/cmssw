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

void testONNXRuntimeCUDA::checkCUDA() { testGPUBackend(Backend::cuda, "CUDAService", cms::cudatest::testDevices); }
