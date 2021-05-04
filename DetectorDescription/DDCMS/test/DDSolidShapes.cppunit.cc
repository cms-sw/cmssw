#include <cppunit/extensions/HelperMacros.h>

#include "DetectorDescription/DDCMS/interface/DDSolidShapes.h"

#include <iostream>

#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

using namespace cms;
using namespace std;

class testDDSolidShapes : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDDSolidShapes);
  CPPUNIT_TEST(checkDDSolidShapes);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override;
  void tearDown() override {}
  void checkDDSolidShapes();

private:
  std::string solidName_;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDDSolidShapes);

void testDDSolidShapes::setUp() { solidName_ = "Trap"; }

void testDDSolidShapes::checkDDSolidShapes() {
  cms::DDSolidShape shape = cms::dd::value(cms::DDSolidShapeMap, solidName_);
  CPPUNIT_ASSERT(shape == cms::DDSolidShape::ddtrap);

  std::string name = cms::dd::name(cms::DDSolidShapeMap, shape);
  CPPUNIT_ASSERT(name == solidName_);

  cms::DDSolidShape invalidShape = cms::dd::value(cms::DDSolidShapeMap, "Blah Blah Blah");
  CPPUNIT_ASSERT(invalidShape == cms::DDSolidShape::dd_not_init);

  std::string invalidName = cms::dd::name(cms::DDSolidShapeMap, invalidShape);
  CPPUNIT_ASSERT(invalidName == std::string("Solid not initialized"));
}
