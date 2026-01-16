/*
 *  eventsetupscontroller_t.cc
 */

#include "cppunit/extensions/HelperMacros.h"
#include "FWCore/Framework/interface/EventSetupsController.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include <string>
#include <vector>

namespace {
  edm::ActivityRegistry activityRegistry;
}

class TestEventSetupsController : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestEventSetupsController);

  CPPUNIT_TEST(constructorTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}

  void constructorTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestEventSetupsController);

void TestEventSetupsController::constructorTest() {
  edm::eventsetup::EventSetupsController esController;

  CPPUNIT_ASSERT(esController.mustFinishConfiguration() == true);

  edm::ParameterSet pset;
  std::vector<std::string> emptyVStrings;
  pset.addParameter<std::vector<std::string> >("@all_esprefers", emptyVStrings);
  pset.addParameter<std::vector<std::string> >("@all_essources", emptyVStrings);
  pset.addParameter<std::vector<std::string> >("@all_esmodules", emptyVStrings);

  esController.makeProvider(pset, &activityRegistry);
}
