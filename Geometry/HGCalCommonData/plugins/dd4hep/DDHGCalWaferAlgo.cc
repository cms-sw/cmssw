#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"

//#define EDM_ML_DEBUG
using namespace cms_units::operators;

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);

  // Header section of original DDHGCalWafer.h
  const auto& cellSize = args.value<double>("CellSize");                         // Cell Size
  const auto& cellType = args.value<int>("CellType");                            // Type (1 fine; 2 coarse)
  const auto& childNames = args.value<std::vector<std::string> >("ChildNames");  // Names of children
  const auto& positionX = args.value<std::vector<int> >("PositionX");            // Position in X
  const auto& positionY = args.value<std::vector<int> >("PositionY");            // Position in Y
  const auto& angles = args.value<std::vector<double> >("Angles");               // Rotation angle
  const auto& detectorType = args.value<std::vector<int> >("DetectorType");      // Detector type
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << childNames.size() << " children: " << childNames[0] << "; " << childNames[1]
                                << " positioned " << positionX.size() << " times with cell size " << cellSize;
  for (unsigned int k = 0; k < positionX.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] x " << positionX[k] << " y " << positionY[k] << " angle "
                                  << angles[k] << " detector " << detectorType[k];

  std::string idName = args.parentName();  // Name of the "parent" volume.
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferAlgo debug: Parent " << idName << " NameSpace " << ns.name();
#endif

  dd4hep::Volume mother = ns.volume(args.parentName());
  double dx = 0.5 * cellSize;
  double dy = 0.5 * dx * tan(30._deg);

  for (unsigned int k = 0; k < positionX.size(); ++k) {
    std::string name(childNames[detectorType[k]]);
    name = ns.prepend(name);
    dd4hep::Rotation3D rotation;
    if (angles[k] != 0) {
      double phi = convertDegToRad(angles[k]);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferAlgo: Creating new rotation \t90, " << angles[k] << ", 90, "
                                    << (angles[k] + 90) << ", 0, 0";
#endif
      rotation = cms::makeRotation3D(90._deg, phi, 90._deg, (90._deg + phi), 0, 0);
    }
    double xpos = dx * positionX[k];
    double ypos = dy * positionY[k];
    dd4hep::Position tran(xpos, ypos, 0);
    int copy = HGCalTypes::packCellType6(cellType, k);
    mother.placeVolume(ns.volume(name), copy, dd4hep::Transform3D(rotation, tran));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferAlgo: " << name << " number " << copy << " positioned in " << idName
                                  << " at " << tran << " with " << rotation;
#endif
  }

  return cms::s_executed;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalWaferAlgo, algorithm)
