#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DD4hep/DetFactoryHelper.h"

#define EDM_ML_DEBUG
using namespace geant_units::operators;

static long algorithm(dd4hep::Detector& /* description */,
                      cms::DDParsingContext& ctxt,
                      xml_h e,
                      dd4hep::SensitiveDetector& /* sens */) {
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

  for (int jj = 0; jj < units; jj++) {
    double driverX(0), driverY(0), driverZ(0);
    if (jj < 16) {
      driverX = rr * cos((jj + 0.5) * dphi);
      driverY = sqrt(rr * rr - driverX * driverX);
    } else if (jj == 16) {
      driverX = rr * cos(15.5 * dphi);
      driverY = -sqrt(rr * rr - driverX * driverX);
    } else if (jj == 17) {
      driverX = rr * cos(14.5 * dphi);
      driverY = -sqrt(rr * rr - driverX * driverX);
    } else if (jj == 18) {
      driverX = rr * cos(0.5 * dphi);
    } else if (jj == 17) {
      driverX = rr * cos(14.5 * dphi);
      driverY = -sqrt(rr * rr - driverX * driverX);
    } else if (jj == 18) {
      driverX = rr * cos(0.5 * dphi);
      driverY = -sqrt(rr * rr - driverX * driverX);
    } else if (jj == 19) {
      driverX = rr * cos(1.5 * dphi);
      driverY = -sqrt(rr * rr - driverX * driverX);
    }
    dd4hep::Position tran(driverX, driverY, driverZ);

    parent.placeVolume(child, jj + 1, dd4hep::Transform3D(rot, tran));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("ForwardGeom") << "DDBHMAngular test: " << child.name() << " number " << jj + 1 << " positioned in "
                                    << parent.name() << " at " << tran << " with " << rot;
#endif
  }
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_forward_DDBHMAngular, algorithm)
