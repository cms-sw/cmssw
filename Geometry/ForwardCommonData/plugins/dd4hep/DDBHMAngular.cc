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
  int units = args.value<int>("number");
  double rr = args.value<double>("radius");
  double dphi = args.value<double>("deltaPhi");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("ForwardGeom") << "DDBHMAngular debug: Parameters for positioning-- " << units
                                  << " copies at radius " << convertMmToCm(rr) << " cm with delta(phi) "
                                  << convertRadToDeg(dphi);
#endif
  std::string rotMat = args.value<std::string>("Rotation");
  std::string childName = args.value<std::string>("ChildName");
  dd4hep::Volume parent = ns.volume(args.parentName());
  dd4hep::Volume child = ns.volume(ns.prepend(childName));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("ForwardGeom") << "DDBHMAngular debug: Parent " << parent.name() << "\tChild " << child.name()
                                  << "\tRotation matrix " << rotMat;
#endif
  std::string rotstr = DDSplit(rotMat).first;
  dd4hep::Rotation3D rot;
  if (rotstr != "NULL") {
    rot = ns.rotation(rotMat);
  }
  static const int startpos = 16;
  static const double fac1 = 0.5;
  static const double fac2 = 1.5;
  for (int jj = 0; jj < units; jj++) {
    double driverX(0), driverY(0), driverZ(0);
    if (jj < startpos) {
      driverX = rr * cos((jj + fac1) * dphi);
      driverY = sqrt(rr * rr - driverX * driverX);
    } else if (jj == startpos) {
      driverX = rr * cos((startpos - fac1) * dphi);
      driverY = -sqrt(rr * rr - driverX * driverX);
    } else if (jj == startpos + 1) {
      driverX = rr * cos((startpos - fac2) * dphi);
      driverY = -sqrt(rr * rr - driverX * driverX);
    } else if (jj == startpos + 2) {
      driverX = rr * cos(fac1 * dphi);
      driverY = -sqrt(rr * rr - driverX * driverX);
    } else if (jj == startpos + 3) {
      driverX = rr * cos(fac2 * dphi);
      driverY = -sqrt(rr * rr - driverX * driverX);
    }
    dd4hep::Position tran(driverX, driverY, driverZ);

    parent.placeVolume(child, jj + 1, dd4hep::Transform3D(rot, tran));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("ForwardGeom") << "DDBHMAngular test: " << child.name() << " number " << jj + 1
                                    << " positioned in " << parent.name() << " at " << tran << " with " << rot;
#endif
  }
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_forward_DDBHMAngular, algorithm)
