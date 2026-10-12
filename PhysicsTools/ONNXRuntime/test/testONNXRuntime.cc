#include <cppunit/extensions/HelperMacros.h>

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "testONNXRuntime.h"

using namespace cms::Ort;

class testONNXRuntime : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testONNXRuntime);
  CPPUNIT_TEST(checkCPU);
  CPPUNIT_TEST_SUITE_END();

public:
  void checkCPU();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testONNXRuntime);

void testONNXRuntime::checkCPU() { testModel(Backend::cpu); }
