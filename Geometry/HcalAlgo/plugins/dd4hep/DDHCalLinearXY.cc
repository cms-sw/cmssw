#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DD4hep/DetFactoryHelper.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);
  // Header section of original DDHCalLinearXY.h
  int numberX = args.value<int>("NumberX");      //Slices along X
  int numberY = args.value<int>("NumberY");      //Slices along Y
  double deltaX = args.value<double>("DeltaX");  //Step along X
  double deltaY = args.value<double>("DeltaY");  //Step along Y
  std::vector<double> centre = args.value<std::vector<double> >("Center");
  dd4hep::Volume parent = ns.volume(args.parentName());
  std::vector<std::string> childName = args.value<std::vector<std::string> >("Child");

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalLinearXY: Parent " << parent.name() << " \twith " << childName.size()
                               << " children";
  unsigned int k(0);
  for (auto const& child : childName) {
    edm::LogVerbatim("HCalGeom") << "DDHCalLinearXY: Child[" << k << "] = " << child;
    ++k;
  }
  edm::LogVerbatim("HCalGeom") << "DDHCalLinearXY: Number along X/Y " << numberX << "/" << numberY
                               << "\tDelta along X/Y " << convertCmToMm(deltaX) << "/" << convertCmToMm(deltaY)
                               << "\tCentre (" << convertCmToMm(centre[0]) << ", " << convertCmToMm(centre[1]) << ","
                               << convertCmToMm(centre[2]);
#endif
  double xoff = centre[0] - (numberX - 1) * 0.5 * deltaX;
  double yoff = centre[1] - (numberY - 1) * 0.5 * deltaY;
  unsigned int copy = 0;
  for (int i = 0; i < numberX; ++i) {
    for (int j = 0; j < numberY; ++j) {
      unsigned int k = (childName.size() == 1) ? 0 : copy;
      ++copy;
      if (k < childName.size() && (childName[k] != " " && childName[k] != "Null")) {
        std::string child = childName[k];
        dd4hep::Position tran(xoff + i * deltaX, yoff + j * deltaY, centre[2]);
        parent.placeVolume(ns.volume(child), copy, tran);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalLinearXY: " << child << " number " << copy << " positioned in "
                                     << parent.name() << " at (" << convertCmToMm(xoff + i * deltaX) << ", "
                                     << convertCmToMm(yoff + j * deltaY) << ", " << convertCmToMm(centre[2])
                                     << ") with no rotation";
#endif
      } else {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalLinearXY: no child placed for [" << copy << "]";
#endif
      }
    }
  }
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hcal_DDHCalLinearXY, algorithm)
