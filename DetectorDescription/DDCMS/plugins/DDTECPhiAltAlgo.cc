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
  double startAngle = args.value<double>("StartAngle");                             //Start angle
  double incrAngle = args.value<double>("IncrAngle");                               //Increment in angle
  double radius = args.value<double>("Radius");                                     //Radius
  double zIn = args.value<double>("ZIn");                                           //z position for the even ones
  double zOut = args.value<double>("ZOut");                                         //z position for the odd  ones
  int number = args.value<double>("Number");                                        //Number of copies
  int startCopyNo = args.find("StartCopyNo") ? args.value<int>("StartCopyNo") : 1;  //Start copy number
  int incrCopyNo = args.find("IncrCopyNo") ? args.value<int>("IncrCopyNo") : 1;     //Increment in copy number

  LogDebug("TECGeom") << "debug: Parameters for "
                      << "positioning--"
                      << "\tStartAngle " << convertRadToDeg(startAngle) << "\tIncrAngle " << convertRadToDeg(incrAngle)
                      << "\tZ in/out " << zIn << ", " << zOut << "\tCopy Numbers " << number << " Start/Increment "
                      << startCopyNo << ", " << incrCopyNo;
  LogDebug("TECGeom") << "debug: Parent " << mother.name() << "\tChild " << child.name() << " NameSpace " << ns.name();

  if (number > 0) {
    double theta = 90._deg;
    int copyNo = startCopyNo;
    for (int i = 0; i < number; i++) {
      double phiz = startAngle + i * incrAngle;
      double phix = phiz + 90._deg;
      Rotation3D rotation = makeRotation3D(theta, phix, 0e0, 0e0, theta, phiz);
      Position tran(radius * cos(phiz), radius * sin(phiz), (i % 2 == 0) ? zIn : zOut);
      mother.placeVolume(child, copyNo, Transform3D(rotation, tran));
      LogDebug("TECGeom") << "test: " << child.name() << " number " << copyNo << " positioned in " << mother.name()
                          << " at " << tran << " with " << rotation;
      copyNo += incrCopyNo;
    }
  }
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTECPhiAltAlgo, algorithm)
