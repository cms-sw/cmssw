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
  double eta = args.value<double>("Eta");                         //Eta at which beam is focussed
  double theta = 2.0 * atan(exp(-eta));                           //Corresponding theta value
  double shiftY = args.value<double>("ShiftX");                   //Shift along Y
  double shiftX = args.value<double>("ShiftY");                   //Shift along X
  double zoffset = args.value<double>("Zoffset");                 //Offset in z
  double dist = args.value<double>("Distance");                   //Radial distance
  double tilt = args.value<double>("TiltAngle");                  //Tilt with respect to y-axis
  int copyNumber = args.value<int>("Number");                     //Copy Number
  std::string childName = args.value<std::string>("ChildName");   //Children name
  if (strchr(childName.c_str(), NAMESPACE_SEP) == nullptr)
    childName = idNameSpace + childName;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBZposAlgo: Parameters for position"
                               << "ing--"
                               << " Eta " << eta << "\tTheta " << convertRadToDeg(theta) << "\tShifts "
                               << convertCmToMm(shiftX) << ", " << convertCmToMm(shiftY) << " along x, y "
                               << "axes; \tZoffest " << convertCmToMm(zoffset) << "\tRadial Distance "
                               << convertCmToMm(dist) << "\tTilt angle " << convertRadToDeg(tilt) << "\tcopyNumber "
                               << copyNumber;
  edm::LogVerbatim("HCalGeom") << "DDHCalTBZposAlgo: Parent " << args.parentName() << "\tChild " << childName
                               << " NameSpace " << idNameSpace;
#endif

  dd4hep::Volume mother = ns.volume(args.parentName());
  dd4hep::Volume child = ns.volume(childName);

  // Create the rotation matrices
  double thetax = 90._deg - theta;
  double z = zoffset + dist * tan(thetax);
  double x = shiftX - shiftY * sin(tilt);
  double y = shiftY * cos(tilt);
  dd4hep::Position tran(x, y, z);
  dd4hep::Rotation3D rot;
  if (tilt != 0) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalZposAlgo: Creating a rotation \t90, " << convertRadToDeg(tilt) << ",90,"
                                 << (90 + convertRadToDeg(tilt)) << ", 0, 0";
#endif
    rot = dd4hep::RotationZ(tilt);
  }
  mother.placeVolume(child, copyNumber, dd4hep::Transform3D(rot, tran));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBZposAlgo: " << child.name() << " number " << copyNumber << " positioned in "
                               << mother.name() << " at (" << convertCmToMm(x) << ", " << convertCmToMm(y) << ", "
                               << convertCmToMm(z) << ") with " << rot;
#endif

  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hcal_DDHCalTBZposAlgo, algorithm);
