#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace cms_units::operators;

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& context, xml_h element) {
  cms::DDNamespace ns(context, element, true);
  cms::DDAlgoArguments args(context, element);

  int n = args.value<int>("n");
  int startCopyNo = args.find("startCopyNo") ? args.value<int>("startCopyNo") : 1;
  int incrCopyNo = args.find("incrCopyNo") ? args.value<int>("incrCopyNo") : 1;
  float startAngle = args.value<float>("startAngle");
  float stepAngle = args.value<float>("stepAngle");
  float zoffset = args.value<float>("zoffset");
  std::string rotns = args.value<std::string>("RotNameSpace");
  dd4hep::Volume mother = ns.volume(args.parentName());
  std::string childName = args.value<std::string>("ChildName");
  childName = ns.prepend(childName);
  dd4hep::Volume child = ns.volume(childName);

  edm::LogVerbatim("MuonGeom") << "debug: Parameters for positioning:: n " << n << " Start, Step "
                               << convertRadToDeg(startAngle) << ", " << convertRadToDeg(stepAngle) << ", zoffset "
                               << zoffset << ", RotNameSpace " << rotns;
  edm::LogVerbatim("MuonGeom") << "debug: Parent " << mother.name() << "\tChild " << child.name() << " NameSpace "
                               << ns.name();

  double phi = startAngle;
  int copyNo = startCopyNo;

  for (int i = 0; i < n; ++i) {
    double phitmp = phi;
    if (phitmp >= 2._pi)
      phitmp -= 2._pi;
    dd4hep::Rotation3D rotation = cms::makeRotation3D(90._deg, phitmp, 90._deg, 90._deg + phitmp, 0., 0.);
    std::string rotstr = ns.nsName(child.name()) + std::to_string(phitmp * 10.);
    auto irot = context.rotations.find(ns.prepend(rotstr));
    if (irot != context.rotations.end()) {
      rotation = ns.rotation(ns.prepend(rotstr));
    }
    dd4hep::Position tran(0., 0., zoffset);
    mother.placeVolume(child, copyNo, dd4hep::Transform3D(rotation, tran));
    edm::LogVerbatim("MuonGeom") << "test " << child.name() << " number " << copyNo << " positioned in "
                                 << mother.name() << " at " << tran << " with " << rotstr << ": " << rotation;
    phi += stepAngle;
    copyNo += incrCopyNo;
  }
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_muon_DDMuonAngular, algorithm)
