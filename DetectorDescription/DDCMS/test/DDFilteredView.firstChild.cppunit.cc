#include <cppunit/extensions/HelperMacros.h>

#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DD4hep/Detector.h"

#include <string>
#include <memory>

#include "cppunit/TestAssert.h"
#include "cppunit/TestFixture.h"

using namespace cms;
using namespace std;

class testDDFilteredViewFirstChild : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDDFilteredViewFirstChild);
  CPPUNIT_TEST(checkFilteredViewFirstChild);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override;
  void tearDown() override {}
  void checkFilteredViewFirstChild();

private:
  string fileName_;
  vector<double> refdattl_;
  vector<string> refsattl_;
  vector<double> refdLongFL_;
  vector<string> refsLongFL_;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDDFilteredViewFirstChild);

void testDDFilteredViewFirstChild::setUp() {
  fileName_ = edm::FileInPath("Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2021.xml").fullPath();
  refdattl_ = {0.000809653, 0.000713002, 0.000654918, 0.000602767, 0.000566295, 0.000541647, 0.000516174, 0.000502512,
               0.000504225, 0.000506212, 0.000506275, 0.000487621, 0.000473034, 0.000454002, 0.000442383, 0.000441043,
               0.000443609, 0.000433124, 0.000440188, 0.000435257, 0.000439224, 0.000431385, 0.00041707,  0.000415677,
               0.000408389, 0.000400293, 0.000400989, 0.000395417, 0.00038936,  0.000383942};
  refsattl_ = {"0.8096535E-03/cm", "0.7130018E-03/cm", "0.6549183E-03/cm", "0.6027666E-03/cm", "0.5662951E-03/cm",
               "0.5416475E-03/cm", "0.5161745E-03/cm", "0.5025120E-03/cm", "0.5042249E-03/cm", "0.5062117E-03/cm",
               "0.5062750E-03/cm", "0.4876212E-03/cm", "0.4730343E-03/cm", "0.4540021E-03/cm", "0.4423832E-03/cm",
               "0.4410428E-03/cm", "0.4436095E-03/cm", "0.4331241E-03/cm", "0.4401879E-03/cm", "0.4352570E-03/cm",
               "0.4392237E-03/cm", "0.4313847E-03/cm", "0.4170704E-03/cm", "0.4156771E-03/cm", "0.4083890E-03/cm",
               "0.4002930E-03/cm", "0.4009888E-03/cm", "0.3954170E-03/cm", "0.3893599E-03/cm", "0.3839422E-03/cm"};
  refdLongFL_ = {
      227.993, 237.122, 241.701, 256.48, 266.754, 275.988, 276.982, 284.989, 286.307, 290.478, 290.5, 292, 295.5};
  refsLongFL_ = {"227.9925651*cm",
                 "237.1215213*cm",
                 "241.7005445*cm",
                 "256.47981*cm",
                 "266.7540042*cm",
                 "275.987715*cm",
                 "276.9823529*cm",
                 "284.9889299*cm",
                 "286.3065327*cm",
                 "290.4779412*cm",
                 "290.5*cm",
                 "292.0*cm",
                 "295.5*cm"};
}

void testDDFilteredViewFirstChild::checkFilteredViewFirstChild() {
  unique_ptr<DDDetector> det = make_unique<DDDetector>("DUMMY", fileName_);
  DDFilteredView fview(det.get(), det->description()->worldVolume());
  const cms::DDSpecParRegistry& mypar = det->specpars();
  std::string attribute{"CMSCutsRegion"};
  cms::DDSpecParRefs ref;
  mypar.filter(ref, attribute, "MuonChamber");
  fview.mergedSpecifics(ref);

  fview.firstChild();
  std::cout << fview.name() << " is a " << cms::dd::name(cms::DDSolidShapeMap, fview.shape()) << "\n";
  std::cout << "Full path to it is " << fview.path() << "\n";
  auto copyNos = fview.copyNos();
  if (dd4hep::isA<dd4hep::Box>(fview.solid()))
    cout << "It's a Box\n";

  std::cout << fview.name() << " is a " << cms::dd::name(cms::DDSolidShapeMap, fview.shape()) << "\n";
  do {
    std::cout << fview.path() << "\n";
  } while (fview.nextChild());
  std::cout << "Current node is:\n" << fview.path() << "\n";
}
