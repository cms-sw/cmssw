#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);
  // Header section
  std::string idNameSpace = static_cast<std::string>(ns.name());  //Namespace of this and ALL sub-parts
  double radius = args.value<double>("Radius");                   //Pointing distance from front surface
  double offset = args.value<double>("Offset");                   //Offset along Z
  double dx = args.value<double>("Dx");                           //Half size along x
  double dz = args.value<double>("Dz");                           //Half size along z
  double angwidth = args.value<double>("AngWidth");               //Angular width
  int iaxis = args.value<int>("Axis");                            //Axis of rotation
  std::vector<std::string> names = args.value<std::vector<std::string> >("Names");  //Names for rotation matrices
  std::string idName = args.value<std::string>("ChildName");                        //Children name
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalXtalAlgo::Parameters for positioning:"
                               << " Axis " << iaxis << "\tRadius " << convertCmToMm(radius) << "\tOffset " << offset
                               << "\tDx " << convertCmToMm(dx) << "\tDz " << convertCmToMm(dz) << "\tAngWidth "
                               << convertRadToDeg(angwidth) << "\tNumbers " << names.size();
  for (unsigned int i = 0; i < names.size(); i++)
    edm::LogVerbatim("HCalGeom") << "\tnames[" << i << "] = " << names[i];
  edm::LogVerbatim("HCalGeom") << "DDHCalXtalAlgo: Parent " << args.parentName() << "\tChild " << idName
                               << " NameSpace " << idNameSpace;
#endif

  dd4hep::Volume parent = ns.volume(args.parentName());

  double theta[3], phi[3], pos[3];
  phi[0] = 0;
  phi[1] = 90._deg;
  theta[1 - iaxis] = 90._deg;
  pos[1 - iaxis] = 0;
  int number = (int)(names.size());
  for (int i = 0; i < number; i++) {
    double angle = 0.5 * angwidth * (2 * i + 1 - number);
    theta[iaxis] = 90._deg + angle;
    if (angle > 0) {
      theta[2] = angle;
      phi[2] = iaxis * 90._deg;
    } else {
      theta[2] = -angle;
      phi[2] = (2 - 3 * iaxis) * 90._deg;
    }
    pos[iaxis] = angle * (dz + radius);
    pos[2] = dx * std::abs(sin(angle)) + offset;

    if (strchr(idName.c_str(), NAMESPACE_SEP) == nullptr)
      idName = idNameSpace + idName;
    dd4hep::Volume glog = ns.volume(idName);
    dd4hep::Position tran(pos[0], pos[1], pos[2]);
    dd4hep::Rotation3D rotation;

    static const double tol = 0.01_deg;  // 0.01 degree
    if (std::abs(angle) > tol) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalXtalAlgo: Creating a rotation \t" << convertRadToDeg(theta[0]) << ","
                                   << convertRadToDeg(phi[0]) << "," << convertRadToDeg(theta[1]) << ","
                                   << convertRadToDeg(phi[1]) << "," << convertRadToDeg(theta[2]) << ","
                                   << convertRadToDeg(phi[2]);
#endif
      rotation = cms::makeRotation3D(theta[0], phi[0], theta[1], phi[1], theta[2], phi[2]);
    }
    parent.placeVolume(glog, i + 1, dd4hep::Transform3D(rotation, tran));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalXtalAlgo: " << glog.name() << " number " << i + 1 << " positioned in "
                                 << parent.name() << " at " << tran << " with " << rotation;
#endif
  }

  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hcal_DDHCalXtalAlgo, algorithm);
