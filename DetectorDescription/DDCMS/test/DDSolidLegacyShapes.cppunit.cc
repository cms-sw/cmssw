#include <cppunit/extensions/HelperMacros.h>

#include "DetectorDescription/DDCMS/interface/DDSolidShapes.h"

#include <iostream>

#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

using namespace cms;
using namespace std;

class testDDSolidLegacyShapes : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDDSolidLegacyShapes);
  CPPUNIT_TEST(checkDDSolidLegacyShapes);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override;
  void tearDown() override {}
  void checkDDSolidLegacyShapes();

private:
  std::string solidName_;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDDSolidLegacyShapes);

void testDDSolidLegacyShapes::setUp() { solidName_ = "Trap"; }

void testDDSolidLegacyShapes::checkDDSolidLegacyShapes() {
  cms::DDSolidShape shape = cms::dd::value(cms::DDSolidShapeMap, solidName_);
  CPPUNIT_ASSERT(shape == cms::DDSolidShape::ddtrap);

  std::string name = cms::dd::name(cms::DDSolidShapeMap, shape);
  CPPUNIT_ASSERT(name == solidName_);

  cms::DDSolidShape invalidShape = cms::dd::value(cms::DDSolidShapeMap, "Blah Blah Blah");
  CPPUNIT_ASSERT(invalidShape == cms::DDSolidShape::dd_not_init);

  std::string invalidName = cms::dd::name(cms::DDSolidShapeMap, invalidShape);
  CPPUNIT_ASSERT(invalidName == std::string("Solid not initialized"));

  LegacySolidShape invalidLegacyShape = cms::dd::value(cms::LegacySolidShapeMap, invalidShape);
  CPPUNIT_ASSERT(invalidLegacyShape == LegacySolidShape::dd_not_init);

  LegacySolidShape legacyBox = cms::dd::value(cms::LegacySolidShapeMap, cms::DDSolidShape::ddbox);
  CPPUNIT_ASSERT(legacyBox == LegacySolidShape::ddbox);

  LegacySolidShape legacyTubs = cms::dd::value(cms::LegacySolidShapeMap, cms::DDSolidShape::ddtubs);
  CPPUNIT_ASSERT(legacyTubs == LegacySolidShape::ddtubs);

  LegacySolidShape legacyTrap = cms::dd::value(cms::LegacySolidShapeMap, cms::DDSolidShape::ddtrap);
  CPPUNIT_ASSERT(legacyTrap == LegacySolidShape::ddtrap);

  LegacySolidShape legacyCons = cms::dd::value(cms::LegacySolidShapeMap, cms::DDSolidShape::ddcons);
  CPPUNIT_ASSERT(legacyCons == LegacySolidShape::ddcons);

  LegacySolidShape legacyPcon = cms::dd::value(cms::LegacySolidShapeMap, cms::DDSolidShape::ddpolycone);
  CPPUNIT_ASSERT(legacyPcon == LegacySolidShape::ddpolycone_rz);

  LegacySolidShape legacyPolyhedra = cms::dd::value(cms::LegacySolidShapeMap, cms::DDSolidShape::ddpolyhedra);
  CPPUNIT_ASSERT(legacyPolyhedra == LegacySolidShape::ddpolyhedra_rz);

  LegacySolidShape legacyTorus = cms::dd::value(cms::LegacySolidShapeMap, cms::DDSolidShape::ddtorus);
  CPPUNIT_ASSERT(legacyTorus == LegacySolidShape::ddtorus);

  LegacySolidShape legacyUnion = cms::dd::value(cms::LegacySolidShapeMap, cms::DDSolidShape::ddunion);
  CPPUNIT_ASSERT(legacyUnion == LegacySolidShape::ddunion);

  LegacySolidShape legacySubtraction = cms::dd::value(cms::LegacySolidShapeMap, cms::DDSolidShape::ddsubtraction);
  CPPUNIT_ASSERT(legacySubtraction == LegacySolidShape::ddsubtraction);

  LegacySolidShape legacyIntersection = cms::dd::value(cms::LegacySolidShapeMap, cms::DDSolidShape::ddintersection);
  CPPUNIT_ASSERT(legacyIntersection == LegacySolidShape::ddintersection);

  LegacySolidShape legacyShapeless = cms::dd::value(cms::LegacySolidShapeMap, cms::DDSolidShape::ddshapeless);
  CPPUNIT_ASSERT(legacyShapeless == LegacySolidShape::ddshapeless);

  LegacySolidShape legacyPseudotrap = cms::dd::value(cms::LegacySolidShapeMap, cms::DDSolidShape::ddpseudotrap);
  CPPUNIT_ASSERT(legacyPseudotrap == LegacySolidShape::ddpseudotrap);

  LegacySolidShape legacyTrunctubs = cms::dd::value(cms::LegacySolidShapeMap, cms::DDSolidShape::ddtrunctubs);
  CPPUNIT_ASSERT(legacyTrunctubs == LegacySolidShape::ddtrunctubs);

  LegacySolidShape legacySphere = cms::dd::value(cms::LegacySolidShapeMap, cms::DDSolidShape::ddsphere);
  CPPUNIT_ASSERT(legacySphere == LegacySolidShape::ddsphere);

  LegacySolidShape legacyEllipticaltube = cms::dd::value(cms::LegacySolidShapeMap, cms::DDSolidShape::ddellipticaltube);
  CPPUNIT_ASSERT(legacyEllipticaltube == LegacySolidShape::ddellipticaltube);

  LegacySolidShape legacyCuttubs = cms::dd::value(cms::LegacySolidShapeMap, cms::DDSolidShape::ddcuttubs);
  CPPUNIT_ASSERT(legacyCuttubs == LegacySolidShape::ddcuttubs);

  LegacySolidShape legacyExtrudedpolygon =
      cms::dd::value(cms::LegacySolidShapeMap, cms::DDSolidShape::ddextrudedpolygon);
  CPPUNIT_ASSERT(legacyExtrudedpolygon == LegacySolidShape::ddextrudedpolygon);
}
