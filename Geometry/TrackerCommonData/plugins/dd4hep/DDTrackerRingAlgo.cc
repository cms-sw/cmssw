#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace dd4hep;
using namespace cms;
using namespace cms_units::operators;

namespace {
  long algorithm(Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
    DDNamespace ns(ctxt, e, true);
    DDAlgoArguments args(ctxt, e);
    Volume mother = ns.volume(args.parentName());
    string childName = args.value<string>("ChildName");
    Volume child = ns.volume(childName);

    string parentName = args.parentName();
    int n = args.value<int>("N");
    int startCopyNo = args.value<int>("StartCopyNo");
    int incrCopyNo = args.value<int>("IncrCopyNo");
    double rangeAngle = args.value<double>("RangeAngle");
    double startAngle = args.value<double>("StartAngle");
    double radius = args.value<double>("Radius");
    vector<double> center = args.value<vector<double> >("Center");
    bool isZPlus = args.value<int>("IsZPlus") == 1;
    double tiltAngle = args.value<double>("TiltAngle");
    bool isFlipped = args.value<int>("IsFlipped") == 1;
    double delta = 0.;

    if (abs(rangeAngle - 360.0_deg) < 0.001_deg) {
      delta = rangeAngle / double(n);
    } else {
      if (n > 1) {
        delta = rangeAngle / double(n - 1);
      } else {
        delta = 0.;
      }
    }

    LogDebug("TrackerGeom") << "DDTrackerRingAlgo debug: Parameters for position"
                            << "ing:: n " << n << " Start, Range, Delta " << convertRadToDeg(startAngle) << " "
                            << convertRadToDeg(rangeAngle) << " " << convertRadToDeg(delta) << " Radius " << radius
                            << " Centre " << center[0] << ", " << center[1] << ", " << center[2];

    LogDebug("TrackerGeom") << "DDTrackerRingAlgo debug: Parent " << parentName << "\tChild " << childName
                            << " NameSpace " << ns.name();

    Rotation3D flipMatrix, tiltMatrix, phiRotMatrix, globalRotMatrix;  // Identity matrix

    // flipMatrix calculus
    if (isFlipped) {
      LogDebug("TrackerGeom") << "DDTrackerRingAlgo test: Creating a new rotation: "
                              << "\t90., 180., "
                              << "90., 90., "
                              << "180., 0.";
      flipMatrix = makeRotation3D(90._deg, 180._deg, 90._deg, 90._deg, 180._deg, 0._deg);
    }
    // tiltMatrix calculus
    if (isZPlus) {
      LogDebug("TrackerGeom") << "DDTrackerRingAlgo test: Creating a new rotation: "
                              << "\t90., 90., " << convertRadToDeg(tiltAngle) << ", 180., "
                              << convertRadToDeg(90._deg - tiltAngle) << ", 0.";
      tiltMatrix = makeRotation3D(90._deg, 90._deg, tiltAngle, 180._deg, 90._deg - tiltAngle, 0._deg);
      if (isFlipped) {
        tiltMatrix *= flipMatrix;
      }
    } else {
      LogDebug("TrackerGeom") << "DDTrackerRingAlgo test: Creating a new rotation: "
                              << "\t90., 90., " << convertRadToDeg(tiltAngle) << ", 0., "
                              << convertRadToDeg(90._deg + tiltAngle) << ", 0.";
      tiltMatrix = makeRotation3D(90._deg, 90._deg, tiltAngle, 0._deg, 90._deg + tiltAngle, 0._deg);
      if (isFlipped) {
        tiltMatrix *= flipMatrix;
      }
    }

    // Loops for all phi values
    double theta = 90._deg;
    int copy = startCopyNo;
    double phi = startAngle;

    for (int i = 0; i < n; ++i) {
      // phiRotMatrix calculus
      double phix = phi;
      double phiy = phix + 90._deg;
      if (phix != 0.) {
        LogDebug("TrackerGeom") << "DDTrackerRingAlgo test: Creating a new rotation: "
                                << "\t90., " << convertRadToDeg(phix) << ", 90.," << convertRadToDeg(phiy)
                                << ", 0., 0.";
        phiRotMatrix = makeRotation3D(theta, phix, theta, phiy, 0., 0.);
      }

      globalRotMatrix = phiRotMatrix * tiltMatrix;

      // translation def
      double xpos = radius * cos(phi) + center[0];
      double ypos = radius * sin(phi) + center[1];
      double zpos = center[2];
      Position tran(xpos, ypos, zpos);

      // Positions child with respect to parent
      mother.placeVolume(child, copy, Transform3D(globalRotMatrix, tran));
      LogDebug("TrackerGeom") << "DDTrackerRingAlgo test " << child.data()->GetName() << " number " << copy
                              << " positioned in " << mother.data()->GetName() << " at " << tran << " with "
                              << globalRotMatrix;

      copy += incrCopyNo;
      phi += delta;
    }
    return s_executed;
  }
}  // namespace
// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTrackerRingAlgo, ::algorithm)
