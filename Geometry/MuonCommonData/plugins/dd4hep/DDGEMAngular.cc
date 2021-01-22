#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace cms_units::operators;

//#define EDM_ML_DEBUG

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);

  // Header section of original DDGEMAngular.h
  float startAngle = args.value<float>("startAngle");
  float stepAngle = args.value<float>("stepAngle");
  int invert = args.value<int>("invert");
  double rPos = args.value<double>("rPosition");
  double zoffset = args.value<double>("zoffset");
  int n = args.value<int>("n");
  int startCopyNo = args.value<int>("startCopyNo");
  int incrCopyNo = args.value<int>("incrCopyNo");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "DDGEMAngular: Parameters for positioning-- " << n << " copies in steps of "
                               << convertRadToDeg(stepAngle) << " from " << convertRadToDeg(startAngle)
                               << " (inversion flag " << invert << ") \trPos " << rPos << " Zoffest " << zoffset
                               << "\tStart and inremental "
                               << "copy nos " << startCopyNo << ", " << incrCopyNo;
#endif
  std::string childName = args.value<std::string>("ChildName");
  childName = ns.prepend(childName);
  std::string parentName = args.parentName();
  parentName = ns.prepend(parentName);
  dd4hep::Volume parent = ns.volume(parentName);
  dd4hep::Volume child = ns.volume(childName);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "DDGEMAngular: Parent " << parentName << "\tChild " << child.name() << "\tNameSpace "
                               << ns.name();
#endif

  // Now position child in mother *n* times
  double phi = startAngle;
  int copyNo = startCopyNo;
  double thetax = 90.0_deg;
  double thetay = invert == 0 ? 0.0 : 180.0_deg;
  for (int ii = 0; ii < n; ii++) {
    double phiz = phi;
    if (phiz >= 2._pi)
      phiz -= 2._pi;
    double phix = invert == 0 ? (90.0_deg + phiz) : (-90.0_deg + phiz);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MuonGeom") << "DDGEMAngular: Creating a rotation " << convertRadToDeg(thetax) << ", "
                                 << convertRadToDeg(phix) << ", " << convertRadToDeg(thetay) << ", 0, "
                                 << convertRadToDeg(thetax) << ", " << convertRadToDeg(phiz);
#endif
    dd4hep::Rotation3D rotation = cms::makeRotation3D(thetax, phix, thetay, 0., thetax, phiz);
    dd4hep::Position tran(rPos * cos(phiz), rPos * sin(phiz), zoffset);
    parent.placeVolume(child, copyNo, dd4hep::Transform3D(rotation, tran));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("MuonGeom") << "DDGEMAngular: " << child.name() << " number " << copyNo << " positioned in "
                                 << parentName << " at " << tran << " with " << rotation;
#endif
    phi += stepAngle;
    copyNo += incrCopyNo;
  }
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_muon_DDGEMAngular, algorithm)
