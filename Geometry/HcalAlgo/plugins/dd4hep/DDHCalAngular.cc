#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DD4hep/DetFactoryHelper.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);
  // Header section of original DDHCalAngular.h
  int n = args.value<int>("n");
  int startCopyNo = args.value<int>("startCopyNo");
  int incrCopyNo = args.value<int>("incrCopyNo");
  double rangeAngle = args.value<double>("rangeAngle");  //Angular range
  double startAngle = args.value<double>("startAngle");  //Start anle
  double shiftX = args.value<double>("shiftX");          //x Shift
  double shiftY = args.value<double>("shiftY");          //y Shift
  double zoffset = args.value<double>("zoffset");        //z offset
  dd4hep::Volume mother = ns.volume(args.parentName());
  std::string childName = args.value<std::string>("ChildName");
  childName = ns.prepend(childName);
  dd4hep::Volume child = ns.volume(childName);

  // Increment
  double dphi = rangeAngle / n;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalAngular: Parameters for positioning::"
                               << " n " << n << " Start, Range, Delta " << convertRadToDeg(startAngle) << " "
                               << convertRadToDeg(rangeAngle) << " " << convertRadToDeg(dphi) << " Shift "
                               << convertCmToMm(shiftX) << ":" << convertCmToMm(shiftY) << "\n Parent " << mother.name()
                               << "\tChild " << child.name() << " NameSpace " << ns.name();
#endif
  int copy = startCopyNo;
  double phix = startAngle;
  for (int ii = 0; ii < n; ++ii) {
    if (phix >= 2._pi)
      phix -= 2._pi;
    else if (phix < 0)
      phix += 2._pi;
    dd4hep::Rotation3D rotation;
    if (std::abs(phix) >= 0.1_deg) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalAngular::Creating a rotation:"
                                   << "\t90., " << convertRadToDeg(phix) << ", 90.," << (90.0 + convertRadToDeg(phix))
                                   << ", 0, 0";
#endif
      rotation = dd4hep::RotationZ(phix);
    }

    double xpos = shiftX * cos(phix) - shiftY * sin(phix);
    double ypos = shiftX * sin(phix) + shiftY * cos(phix);
    dd4hep::Position tran(xpos, ypos, zoffset);
    mother.placeVolume(child, copy, dd4hep::Transform3D(rotation, tran));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalAngular:: " << child.name() << " number " << copy << " positioned in "
                                 << mother.name() << " at (" << convertCmToMm(xpos) << ", " << convertCmToMm(ypos)
                                 << ", " << convertCmToMm(zoffset) << ") with rotation matrix: " << rotation;
#endif
    copy += incrCopyNo;
    phix += dphi;
  }
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hcal_DDHCalAngular, algorithm)
