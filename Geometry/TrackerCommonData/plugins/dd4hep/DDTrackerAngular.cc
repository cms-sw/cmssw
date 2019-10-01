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
  // Header section of original DDTrackerAngular.h
  int n = args.value<int>("N");
  int startCopyNo = args.find("StartCopyNo") ? args.value<int>("StartCopyNo") : 1;
  int incrCopyNo = args.find("IncrCopyNo") ? args.value<int>("IncrCopyNo") : 1;
  double rangeAngle = args.value<double>("RangeAngle");           //Range in angle
  double startAngle = args.value<double>("StartAngle");           //Start anle
  double radius = args.value<double>("Radius");                   //Radius
  vector<double> center = args.value<vector<double> >("Center");  //Phi values
  Volume mother = ns.volume(args.parentName());
  Volume child = ns.volume(args.value<string>("ChildName"));

  double delta = 0e0;  //Increment in phi
  // Code section of original DDTrackerAngular.cc
  if (fabs(rangeAngle - 360.0_deg) < 0.001_deg) {
    delta = rangeAngle / double(n);
  } else if (n > 1) {
    delta = rangeAngle / double(n - 1);
  }

  edm::LogVerbatim("TrackerGeom") << "DDTrackerAngular debug: Parameters for positioning:: n " << n
                                  << " Start, Range, Delta " << convertRadToDeg(startAngle) << " "
                                  << convertRadToDeg(rangeAngle) << " " << convertRadToDeg(delta) << " Radius "
                                  << radius << " Centre " << center[0] << ", " << center[1] << ", " << center[2];
  edm::LogVerbatim("TrackerGeom") << "DDTrackerAngular debug: Parent " << mother.name() << "\tChild " << child.name()
                                  << " NameSpace " << ns.name();

  double theta = 90._deg;
  int copy = startCopyNo;
  double phi = startAngle;
  for (int i = 0; i < n; i++) {
    double phix = phi;
    double phiy = phix + 90._deg;
    double phideg = convertRadToDeg(phix);

    Rotation3D rotation;
    if (phideg != 0) {
      string rotstr = ns.nsName(child.name()) + std::to_string(phideg * 10.);
      auto irot = ctxt.rotations.find(ns.prepend(rotstr));
      if (irot != ctxt.rotations.end()) {
        rotation = ns.rotation(ns.prepend(rotstr));
      } else {
        edm::LogVerbatim("TrackerGeom") << "DDTrackerAngular Creating a new "
                                        << "rotation: " << rotstr << "\t90., " << convertRadToDeg(phix) << ", 90.,"
                                        << convertRadToDeg(phiy) << ", 0, 0";
        RotationZYX rot;
        rotation = makeRotation3D(theta, phix, theta, phiy, 0., 0.);
      }
    }

    double xpos = radius * cos(phi) + center[0];
    double ypos = radius * sin(phi) + center[1];
    double zpos = center[2];
    Position tran(xpos, ypos, zpos);
    mother.placeVolume(child, copy, Transform3D(rotation, tran));
    edm::LogVerbatim("TrackerGeom") << "DDTrackerAngular test " << child.name() << " number " << copy
                                    << " positioned in " << mother.name() << " at " << tran << " with " << rotation;
    copy += incrCopyNo;
    phi += delta;
  }
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTrackerAngular, algorithm)
