#include "DataFormats/GeometryVector/interface/Phi.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DD4hep/DetFactoryHelper.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);
  double startAngle = args.value<double>("startAngle");
  double stepAngle = args.value<double>("stepAngle");
  double zoffset = args.value<double>("zoffset");
  double roffset = args.value<double>("roffset");
  int n = args.value<int>("n");
  int startCopyNo = args.value<int>("startCopyNo");
  int incrCopyNo = args.value<int>("incrCopyNo");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("ForwardGeom") << "DDTotemAngular: Parameters for positioning-- " << n << " copies in steps of "
                                  << convertRadToDeg(stepAngle) << " from " << convertRadToDeg(startAngle)
                                  << " \tZoffset " << zoffset << " \tRoffset " << roffset
                                  << "\tStart and inremental copy nos " << startCopyNo << ", " << incrCopyNo;
#endif
  std::string childName = args.value<std::string>("ChildName");
  dd4hep::Volume parent = ns.volume(args.parentName());
  dd4hep::Volume child = ns.volume(ns.prepend(childName));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("ForwardGeom") << "DDTotemAngular debug: Parent " << parent.name() << "\tChild " << child.name();
#endif

  double phi = startAngle;
  int copyNo = startCopyNo;

  for (int ii = 0; ii < n; ii++) {
    Geom::Phi0To2pi<double> phitmp = phi;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("ForwardGeom") << "DDTotemAngular: Creating a new rotation \t90, "
                                    << convertRadToDeg(phitmp + 90._deg) << ", 0, 0, 90, " << convertRadToDeg(phitmp);
#endif
    dd4hep::Rotation3D rotation = cms::makeRotation3D(90._deg, 90._deg + phitmp, 0., 0., 90._deg, phitmp);
    dd4hep::Position tran(roffset * cos(phi), roffset * sin(phi), zoffset);

    parent.placeVolume(child, copyNo, dd4hep::Transform3D(rotation, tran));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("ForwardGeom") << "DDTotemAngular: " << child.name() << " number " << copyNo << " positioned in "
                                    << parent.name() << " at " << tran << " with " << rotation;
#endif
    phi += stepAngle;
    copyNo += incrCopyNo;
  }
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_forward_DDTotemAngular, algorithm)
