/*
 * DDHGCalWafer8.cc
 *
 *  Created on: 02-Jul-2019
 *      Author: rsehgal
 */

#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"

//#define EDM_ML_DEBUG

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);
  std::string motherName = args.parentName();
  auto waferSize = args.value<double>("WaferSize");
  auto waferT = args.value<double>("WaferThick");
  auto waferSepar = args.value<double>("SensorSeparation");
  auto nCells = args.value<int>("NCells");
  auto cellType = args.value<int>("CellType");
  auto material = args.value<std::string>("Material");
  auto cellNames = args.value<std::vector<std::string>>("CellNames");

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWafer8: Wafer 2r " << waferSize << " T " << waferT << " Half Separation "
                                << waferSepar << " Cells/Wafer " << nCells << " Cell Type " << cellType << " Material "
                                << material << " Names " << motherName << " NameSpace " << ns.name() << " # of cells "
                                << cellNames.size();
  for (unsigned int k = 0; k < cellNames.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "DDHGCalWafer8: Cell[" << k << "] " << cellNames[k];
#endif

  static const double sqrt3 = std::sqrt(3.0);
  double rM = 0.5 * (waferSize + waferSepar);
  double RM2 = rM / sqrt3;
  double R = waferSize / (3.0 * nCells);
  double r = 0.5 * R * sqrt3;

  // Mother Wafer
  std::vector<double> xM = {rM, 0, -rM, -rM, 0, rM};
  std::vector<double> yM = {RM2, 2 * RM2, RM2, -RM2, -2 * RM2, -RM2};
  std::vector<double> zw = {-0.5 * waferT, 0.5 * waferT};
  std::vector<double> zx(2, 0), zy(2, 0), scale(2, 1.0);

  dd4hep::Material matter = ns.material(material);

  dd4hep::Solid solid = dd4hep::ExtrudedPolygon(xM, yM, zw, zx, zy, scale);
  ns.addSolidNS(ns.prepend(motherName), solid);
  dd4hep::Volume glog = dd4hep::Volume(solid.name(), solid, matter);
  ns.addVolumeNS(glog);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalWafer8: " << solid.name() << " extruded polygon made of " << material
                                << " z|x|y|s (0) " << zw[0] << ":" << zx[0] << ":" << zy[0] << ":" << scale[0]
                                << " z|x|y|s (1) " << zw[1] << ":" << zx[1] << ":" << zy[1] << ":" << scale[1]
                                << " and " << xM.size() << " edges";
  for (unsigned int k = 0; k < xM.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << xM[k] << ":" << yM[k];
#endif

  dd4hep::Rotation3D rotation;
#ifdef EDM_ML_DEBUG
  int counter(0);
#endif
  for (int u = 0; u < 2 * nCells; ++u) {
    for (int v = 0; v < 2 * nCells; ++v) {
      if (((v - u) < nCells) && (u - v) <= nCells) {
#ifdef EDM_ML_DEBUG
        counter++;
#endif
        int n2 = nCells / 2;
        double yp = (u - 0.5 * v - n2) * 2 * r;
        double xp = (1.5 * (v - nCells) + 1.0) * R;
        int cell(0);
        if ((u == 0) && (v == 0))
          cell = 7;
        else if ((u == 0) && (v == nCells - 1))
          cell = 8;
        else if ((u == nCells) && (v == 2 * nCells - 1))
          cell = 9;
        else if ((u == 2 * nCells - 1) && (v == 2 * nCells - 1))
          cell = 10;
        else if ((u == 2 * nCells - 1) && (v == nCells - 1))
          cell = 11;
        else if ((u == nCells) && (v == 0))
          cell = 12;
        else if (u == 0)
          cell = 1;
        else if ((v - u) == (nCells - 1))
          cell = 4;
        else if (v == (2 * nCells - 1))
          cell = 2;
        else if (u == (2 * nCells - 1))
          cell = 5;
        else if ((u - v) == nCells)
          cell = 3;
        else if (v == 0)
          cell = 6;
        dd4hep::Position tran(xp, yp, 0);
        int copy = HGCalTypes::packCellTypeUV(cellType, u, v);
        glog.placeVolume(ns.volume(cellNames[cell]), copy, dd4hep::Transform3D(rotation, tran));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "DDHGCalWafer8: " << cellNames[cell] << " number " << copy << " position in "
                                      << glog.name() << " at " << tran << " with " << rotation;
#endif
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "\nDDHGCalWafer8::Counter : " << counter << "\n===============================\n";
#endif

  return cms::s_executed;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalWafer8, algorithm)
