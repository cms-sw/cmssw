#include <cppunit/extensions/HelperMacros.h>

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "testONNXRuntime.h"

using namespace cms::Ort;

class testONNXRuntime : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testONNXRuntime);
  CPPUNIT_TEST(checkCPU);
  CPPUNIT_TEST(checkMissingService);
  CPPUNIT_TEST_SUITE_END();

public:
  void checkCPU();
  void checkMissingService();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testONNXRuntime);

void testONNXRuntime::checkCPU() {
  edm::ServiceToken token = makeServices();
  edm::ServiceRegistry::Operate operate(token);
  testModel(Backend::cpu);
}

// The ONNXService is required also to run on the CPU: edm::Service throws if it is not available.
void testONNXRuntime::checkMissingService() {
  bool thrown = false;
  try {
    ONNXRuntime::defaultSessionOptions(Backend::cpu);
  } catch (edm::Exception const& e) {
    thrown = true;
    CPPUNIT_ASSERT_EQUAL(edm::errors::NotFound, e.categoryCode());
  }
  CPPUNIT_ASSERT(thrown);
}
