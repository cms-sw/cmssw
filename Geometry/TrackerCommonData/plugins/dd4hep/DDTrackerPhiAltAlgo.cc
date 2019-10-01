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
  Volume mother = ns.volume(args.parentName());
  Volume child = ns.volume(args.childName());
  double tilt = args.value<double>("Tilt");                                         //Tilt of the module
  double startAngle = args.value<double>("StartAngle");                             //offset in phi
  double rangeAngle = args.value<double>("RangeAngle");                             //Maximum range in phi
  double radiusIn = args.value<double>("RadiusIn");                                 //Inner radius
  double radiusOut = args.value<double>("RadiusOut");                               //Outer radius
  double zpos = args.value<double>("ZPosition");                                    //z position
  int number = args.value<int>("Number");                                           //Number of copies
  int startCopyNo = args.find("StartCopyNo") ? args.value<int>("StartCopyNo") : 1;  //Start copy number
  int incrCopyNo = args.find("IncrCopyNo") ? args.value<int>("IncrCopyNo") : 1;     //Increment in copy number

  LogDebug("TrackerGeom") << "Parent " << mother.name() << "\tChild " << child.name() << " NameSpace " << ns.name();
  LogDebug("TrackerGeom") << "Parameters for positioning-- Tilt " << tilt << "\tStartAngle "
                          << convertRadToDeg(startAngle) << "\tRangeAngle " << convertRadToDeg(rangeAngle) << "\tRin "
                          << radiusIn << "\tRout " << radiusOut << "\t ZPos " << zpos << "\tCopy Numbers " << number
                          << " Start/Increment " << startCopyNo << ", " << incrCopyNo;

  if (number > 0) {
    double theta = 90._deg;
    double dphi;
    if (number == 1 || fabs(rangeAngle - 360.0_deg) < 0.001_deg)
      dphi = rangeAngle / number;
    else
      dphi = rangeAngle / (number - 1);
    int copyNo = startCopyNo;

    for (int i = 0; i < number; i++) {
      double phi = startAngle + i * dphi;
      double phix = phi - tilt + 90._deg;
      double phiy = phix + 90._deg;
      double phideg = convertRadToDeg(phix);

      Rotation3D rotation;
      if (phideg != 0) {
        rotation = makeRotation3D(theta, phix, theta, phiy, 0., 0.);
      }

      double xpos, ypos;
      if (i % 2 == 0) {
        xpos = radiusIn * cos(phi);
        ypos = radiusIn * sin(phi);
      } else {
        xpos = radiusOut * cos(phi);
        ypos = radiusOut * sin(phi);
      }
      Position tran(xpos, ypos, zpos);
      /* PlacedVolume pv = */ mother.placeVolume(child, copyNo, Transform3D(rotation, tran));
      LogDebug("TrackerGeom") << "" << child.name() << " number " << copyNo << " positioned in " << mother.name()
                              << " at " << tran << " with " << rotation;
      copyNo += incrCopyNo;
    }
  }
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTrackerPhiAltAlgo, algorithm)
