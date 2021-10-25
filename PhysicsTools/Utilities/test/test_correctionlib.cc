#include <cmath>
#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "correction.h"

class test_correctionlib : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(test_correctionlib);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void checkAll();
};

CPPUNIT_TEST_SUITE_REGISTRATION(test_correctionlib);

void test_correctionlib::checkAll() {
  edm::FileInPath testfile("PhysicsTools/Utilities/test/corrections.json");
  auto cset = correction::CorrectionSet::from_file(testfile.fullPath());
  CPPUNIT_ASSERT(cset->at("test corr"));
  CPPUNIT_ASSERT_THROW(cset->at("nonexistent"), std::out_of_range);
  auto corr = cset->at("test corr");
  CPPUNIT_ASSERT(corr->evaluate({12.0, "blah"}) == 1.1);
  CPPUNIT_ASSERT(corr->evaluate({31.0, "blah3"}) == 0.25 * 31.0 + std::exp(3.1));
}
