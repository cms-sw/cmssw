//Test of the external CICADA model emulation model loading and model unloading
//Developed by Andrew Loeliger, Princeton University, Feb 23, 2023

//We can't test a load of a bad model here, since that is a segfault, not an exception, which is
//OS level and cppunit cannot test against that in any way that qualifies as a success

//TODO: However, it would be good in the future to assure that loading multiple CICADA models at the
//same time have the correct function symbols assigned to each simultaneously
//i.e. CICADA_v1's predict is not overwritten by CICADA_v2's predict if it is loaded later with a
//CICADA_v1 still around

//TODO: might also be nice to have a test for model integrity? Known test cases producing known outputs?
//This may not be appropriate for unit testing however.

#include "ap_fixed.h"
#include "hls4ml/emulator.h"

#include "cppunit/extensions/HelperMacros.h"
#include <memory>
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"

class test_CICADA : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(test_CICADA);
  CPPUNIT_TEST(doModelV1Load);
  CPPUNIT_TEST(doModelV2Load);
  CPPUNIT_TEST(doMultiModelLoad);
  CPPUNIT_TEST_SUITE_END();

public:
  void doModelV1Load();
  void doModelV2Load();
  void doMultiModelLoad();
};

CPPUNIT_TEST_SUITE_REGISTRATION(test_CICADA);

void test_CICADA::doModelV1Load() {
  auto loader = hls4mlEmulator::ModelLoader("CICADAModel_v1");
  auto model = loader.load_model();
}

void test_CICADA::doModelV2Load() {
  auto loader = hls4mlEmulator::ModelLoader("CICADAModel_v2");
  auto model = loader.load_model();
}

void test_CICADA::doMultiModelLoad() {
  auto loader_v1 = hls4mlEmulator::ModelLoader("CICADAModel_v1");
  auto loader_v2 = hls4mlEmulator::ModelLoader("CICADAModel_v2");
  auto model_v1 = loader_v1.load_model();
  auto model_v2 = loader_v2.load_model();
}
