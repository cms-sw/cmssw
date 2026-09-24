#include <cppunit/extensions/HelperMacros.h>

#include "HeterogeneousCore/ROCmUtilities/interface/requireDevices.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "testONNXRuntime.h"

using namespace cms::Ort;

class testONNXRuntimeROCm : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testONNXRuntimeROCm);
  CPPUNIT_TEST(checkROCm);
  CPPUNIT_TEST_SUITE_END();

public:
  void checkROCm();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testONNXRuntimeROCm);

void testONNXRuntimeROCm::checkROCm() { testGPUBackend(Backend::rocm, "ROCmService", cms::rocmtest::testDevices); }
