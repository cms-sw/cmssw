#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace dd4hep;
using namespace cms;
using namespace cms_units::operators;

static long algorithm(Detector& /* description */,
                      cms::DDParsingContext& context,
                      xml_h element,
                      SensitiveDetector& /* sens */) {
  cms::DDNamespace ns(context, element, true);
  DDAlgoArguments args(context, element);

  int n = args.value<int>("n");
  int startCopyNo = args.find("startCopyNo") ? args.value<int>("startCopyNo") : 1;
  int incrCopyNo = args.find("incrCopyNo") ? args.value<int>("incrCopyNo") : 1;
  double startAngle = args.value<double>("startAngle");
  double stepAngle = args.value<double>("stepAngle");
  double zoffset = args.value<double>("zoffset");
  string rotns = args.value<string>("RotNameSpace");
  Volume mother = ns.volume(args.parentName());
  string childName = args.value<string>("ChildName");
  childName = ns.prepend(childName);
  Volume child = ns.volume(childName);

  LogDebug("DDAlgorithm") << "debug: Parameters for positioning:: n " << n << " Start, Step "
                          << convertRadToDeg(startAngle) << " " << convertRadToDeg(stepAngle) << " "
                          << ", zoffset " << zoffset << " "
                          << ", RotNameSpace " << rotns.c_str();
  LogDebug("DDAlgorithm") << "debug: Parent " << mother.name() << "\tChild " << child.name() << " NameSpace "
                          << ns.name();

  double phi = startAngle;
  int copyNo = startCopyNo;

  for (int i = 0; i < n; ++i) {
    double phitmp = phi;
    if (phitmp >= 2._pi)
      phitmp -= 2._pi;
    Rotation3D rotation = makeRotation3D(90._deg, phitmp, 90._deg, 90._deg + phitmp, 0., 0.);
    string rotstr = ns.nsName(child.name()) + std::to_string(phitmp * 10.);
    auto irot = context.rotations.find(ns.prepend(rotstr));
    if (irot != context.rotations.end()) {
      rotation = ns.rotation(ns.prepend(rotstr));
    }
    Position tran(0., 0., zoffset);
    mother.placeVolume(child, copyNo, Transform3D(rotation, tran));
    LogDebug("DDAlgorithm") << "test " << child.name() << " number " << copyNo << " positioned in " << mother.name()
                            << " at " << tran << " with " << rotstr << ": " << rotation;
    phi += stepAngle;
    copyNo += incrCopyNo;
  }
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_muon_DDMuonAngular, algorithm)
