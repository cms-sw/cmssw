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

class testDDFilteredView : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testDDFilteredView);
  CPPUNIT_TEST(checkFilteredView);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() override;
  void tearDown() override {}
  void checkFilteredView();

private:
  string fileName_;
  vector<double> refdattl_;
  vector<string> refsattl_;
  vector<double> refdLongFL_;
  vector<string> refsLongFL_;
};

CPPUNIT_TEST_SUITE_REGISTRATION(testDDFilteredView);

void testDDFilteredView::setUp() {
  fileName_ = edm::FileInPath("DetectorDescription/DDCMS/data/cms-geometry-2021.xml").fullPath();
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

void testDDFilteredView::checkFilteredView() {
  unique_ptr<DDDetector> det = make_unique<DDDetector>("DUMMY", fileName_);
  DDFilteredView fview(det.get(), det->description()->worldVolume());
  fview.next(0);
  std::cout << fview.name() << " is a " << cms::dd::name(cms::DDSolidShapeMap, fview.shape()) << "\n";
  std::cout << "Full path to it is " << fview.path() << "\n";
  auto copyNos = fview.copyNos();
  if (fview.isA<dd4hep::Box>())
    cout << "It's a Box\n";
  fview.parent();
  std::cout << fview.name() << " is a " << cms::dd::name(cms::DDSolidShapeMap, fview.shape()) << "\n";

  std::cout << "All SpecPar names:\n";
  for (auto const n : det->specpars().names())
    std::cout << n << "\n";

  //
  // SpecPar Reference values
  //
  // Name: hf
  // Paths: //HVQX, //HVQF
  // LongFL = 227.9925651*cm, 237.1215213*cm, 241.7005445*cm, 256.47981*cm, 266.7540042*cm, 275.987715*cm, 276.9823529*cm, 284.9889299*cm, 286.3065327*cm, 290.4779412*cm, 290.5*cm, 292.0*cm, 295.5*cm
  // ShortFL = 206.0*cm, 211.8810861*cm, 220.3822464*cm, 235.5520581*cm, 245.6204691*cm, 253.9086538*cm, 255.0117647*cm, 263.0073529*cm, 264.3480392*cm, 268.5*cm, 268.5*cm, 270.0*cm, 273.5*cm
  // Volume = HF
  // lambLim = 300.0, 600.0
  // attl = 0.8096535E-03/cm, 0.7130018E-03/cm, 0.6549183E-03/cm, 0.6027666E-03/cm, 0.5662951E-03/cm, 0.5416475E-03/cm, 0.5161745E-03/cm, 0.5025120E-03/cm, 0.5042249E-03/cm, 0.5062117E-03/cm, 0.5062750E-03/cm, 0.4876212E-03/cm, 0.4730343E-03/cm, 0.4540021E-03/cm, 0.4423832E-03/cm, 0.4410428E-03/cm, 0.4436095E-03/cm, 0.4331241E-03/cm, 0.4401879E-03/cm, 0.4352570E-03/cm, 0.4392237E-03/cm, 0.4313847E-03/cm, 0.4170704E-03/cm, 0.4156771E-03/cm, 0.4083890E-03/cm, 0.4002930E-03/cm, 0.4009888E-03/cm, 0.3954170E-03/cm, 0.3893599E-03/cm, 0.3839422E-03/cm
  // Levels => 4, 5

  cout << "Get attl from hf as double values:\n";
  vector<double> attl = fview.get<vector<double>>("hf", "attl");
  int count(0);
  for (auto const& i : attl) {
    std::cout << "attl " << i << " == " << refdattl_[count] << "\n";
    CPPUNIT_ASSERT(abs(i - refdattl_[count]) < 10e-6);
    count++;
  }

  std::cout << "Get LongFL from hf as double values:\n";
  count = 0;
  std::vector<double> LongFL = fview.get<std::vector<double>>("hf", "LongFL");
  for (auto const& i : LongFL) {
    std::cout << "LongFL " << i << " == " << refdLongFL_[count] << "\n";
    CPPUNIT_ASSERT(abs(i - refdLongFL_[count]) < 10e-2);
    count++;
  }

  std::cout << "Get LongFL from hf as string values:\n";
  count = 0;
  std::vector<std::string> sLongFL = fview.get<std::vector<std::string>>("hf", "LongFL");
  for (auto const& i : sLongFL) {
    std::cout << "LongFL " << i << " == " << refsLongFL_[count] << "\n";
    CPPUNIT_ASSERT(i.compare(refsLongFL_[count]) == 0);
    count++;
  }
}
