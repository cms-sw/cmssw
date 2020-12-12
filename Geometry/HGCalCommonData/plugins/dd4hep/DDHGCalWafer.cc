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
  std::string nsName = static_cast<std::string>(ns.name());
  // Header section of original DDHGCalWafer.h
  double waferSize = args.value<double>("WaferSize");
  int cellType = args.value<int>("CellType");
  int nColumns = args.value<int>("NColumns");
  int nBottomY = args.value<int>("NBottomY");
  std::vector<std::string> childNames = args.value<std::vector<std::string> >("ChildNames");
  std::vector<int> nCellsRow = args.value<std::vector<int> >("NCellsRow");
  std::vector<int> angleEdges = args.value<std::vector<int> >("AngleEdges");
  std::vector<int> detectorType = args.value<std::vector<int> >("DetectorType");
  std::string parentName = args.parentName();
  dd4hep::Volume mother = ns.volume(args.parentName());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << childNames.size() << " children: " << childNames[0] << "; " << childNames[1]
                                << " positioned in " << nCellsRow.size() << " rows and " << nColumns
                                << " columns with lowest column at " << nBottomY << " in mother " << parentName
                                << " of size " << waferSize;
  for (unsigned int k = 0; k < nCellsRow.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] Ncells " << nCellsRow[k] << " Edge rotations " << angleEdges[2 * k]
                                  << ":" << angleEdges[2 * k + 1] << " Type of edge cells " << detectorType[2 * k]
                                  << ":" << detectorType[2 * k + 1];
#endif

  static const int incAlongX = 2;
  static const int incAlongY = 6;
  double dx = 0.5 * waferSize / nColumns;
  double dy = 0.5 * dx * tan(30._deg);
  int ny = nBottomY;
  int kount(0);

  for (unsigned int ir = 0; ir < nCellsRow.size(); ++ir) {
    int nx = 1 - nCellsRow[ir];
    double ypos = dy * ny;
    for (int ic = 0; ic < nCellsRow[ir]; ++ic) {
      std::string name(childNames[0]);
      int irot(0);
      if (ic == 0) {
        name = childNames[detectorType[2 * ir]];
        irot = angleEdges[2 * ir];
      } else if (ic + 1 == nCellsRow[ir]) {
        name = childNames[detectorType[2 * ir + 1]];
        irot = angleEdges[2 * ir + 1];
      }
      dd4hep::Rotation3D rotation;
      if (irot != 0) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalWaferAlgo: Creating "
                                      << "rotation "
                                      << "\t90, " << irot << ", 90, " << (irot + 90) << ", 0, 0";
#endif
        double phix = convertDegToRad(irot);
        double phiy = convertDegToRad(90 + irot);
        rotation = cms::makeRotation3D(90._deg, phix, 90._deg, phiy, 0., 0.);
      }
      std::string namx = nsName + name;
      double xpos = dx * nx;
      nx += incAlongX;
      dd4hep::Position tran(xpos, ypos, 0);
      int copy = HGCalTypes::packCellType6(cellType, kount);
      mother.placeVolume(ns.volume(namx), copy, dd4hep::Transform3D(rotation, tran));
      ++kount;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "DDHGCalWafer: " << name << " number " << copy << " positioned in " << parentName
                                    << " at " << tran << " with " << rotation;
#endif
    }
    ny += incAlongY;
  }
  return cms::s_executed;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalWafer, algorithm)
