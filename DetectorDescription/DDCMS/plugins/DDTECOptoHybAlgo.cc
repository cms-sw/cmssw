#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace dd4hep;
using namespace cms;
using namespace cms_units::operators;

static long algorithm(Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e, SensitiveDetector& /* sens */) {
  cms::DDNamespace ns(ctxt, e, true);
  DDAlgoArguments args(ctxt, e);
  int startCopyNo = args.value<int>("StartCopyNo");
  double rpos = args.value<double>("Rpos");
  double zpos = args.value<double>("Zpos");
  double optoHeight = args.value<double>("OptoHeight");
  double optoWidth = args.value<double>("OptoWidth");
  vector<double> angles = args.value<vector<double> >("Angles");
  Volume child = ns.volume(args.value<string>("ChildName"));
  Volume mother = ns.volume(args.parentName());

  LogDebug("TECGeom") << "Parent " << mother.name() << " Child " << child.name() << " NameSpace " << ns.name();
  LogDebug("TECGeom") << "Height of the Hybrid " << optoHeight << " and Width " << optoWidth << "Rpos " << rpos
                      << " Zpos " << zpos << " StartCopyNo " << startCopyNo << " Number " << angles.size();

  // given r positions are for the lower left corner
  rpos += optoHeight / 2;
  int copyNo = startCopyNo;
  for (double angle : angles) {
    double phix = -angle;
    // given phi positions are for the lower left corner
    phix += asin(optoWidth / 2 / rpos);
    double xpos = rpos * cos(phix);
    double ypos = rpos * sin(phix);
    Position tran(xpos, ypos, zpos);

    Rotation3D rotation;
    double phiy = phix + 90._deg;
    double phideg = convertRadToDeg(phix);
    if (phideg != 0) {
      string rotstr = ns.nsName(child.name()) + std::to_string(phideg * 1000.);
      auto irot = ctxt.rotations.find(ns.prepend(rotstr));
      if (irot != ctxt.rotations.end()) {
        rotation = ns.rotation(ns.prepend(rotstr));
      } else {
        double theta = 90._deg;
        LogDebug("TECGeom") << "test: Creating a new "
                            << "rotation: " << rotstr << "\t90., " << convertRadToDeg(phix) << ", 90.,"
                            << convertRadToDeg(phiy) << ", 0, 0";
        rotation = makeRotation3D(theta, phix, theta, phiy, 0., 0.);
      }
    }
    mother.placeVolume(child, copyNo, Transform3D(rotation, tran));
    LogDebug("TECGeom") << "test " << child.name() << " number " << copyNo << " positioned in " << mother.name()
                        << " at " << tran << " with " << rotation;
    copyNo++;
  }
  LogDebug("TECGeom") << "<<== End of DDTECOptoHybAlgo construction ...";
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTECOptoHybAlgo, algorithm)
