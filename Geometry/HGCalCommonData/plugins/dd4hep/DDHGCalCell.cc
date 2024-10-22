#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "DetectorDescription/DDCMS/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

//#define EDM_ML_DEBUG

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);
  // Header section of original DDHGCalCell.h
  double waferSize = args.value<double>("WaferSize");                // size of wafer
  double waferSeparation = args.value<double>("WaferSeparation");    // Wafer Separation
  int addWaferSeparation = args.value<int>("AddWaferSeparation");    // Use wafer separation
  double waferT = args.value<double>("WaferThick");                  // Thickness of wafer
  double cellT = args.value<double>("CellThick");                    // Thickness of depletion layer
  int nCells = args.value<int>("NCells");                            // Number of cells
  int posSens = args.value<int>("PosSensitive");                     // Position of delpletion layer within wafer
  std::string material = args.value<std::string>("Material");        // Material
  std::string fullCN = args.value<std::string>("FullCell");          // Name of the full cell
  std::string fullSensN = args.value<std::string>("FullSensitive");  // Name of the sensitive part for a full cell
  std::vector<std::string> truncCN =
      args.value<std::vector<std::string> >("TruncatedCell");  // Names of truncated cells
  std::vector<std::string> truncSensN =
      args.value<std::vector<std::string> >("TruncatedSensitive");  // Names of the sensitive parts for truncated cells
  std::vector<std::string> extenCN = args.value<std::vector<std::string> >("ExtendedCell");  // Names of extended cells
  std::vector<std::string> extenSensN =
      args.value<std::vector<std::string> >("ExtendedSensitive");  // Names of the sensitive parts for extended cells
  std::vector<std::string> cornrCN = args.value<std::vector<std::string> >("CornerCell");  // Names of the corner cells
  std::vector<std::string> cornrSensN =
      args.value<std::vector<std::string> >("CornerSensitive");  // Names of the sensitive parts for corner cells

  if ((truncCN.size() != truncSensN.size()) || (extenCN.size() != extenSensN.size()) ||
      (cornrCN.size() != cornrSensN.size())) {
    edm::LogError("HGCalGeom") << "The number of cells & sensitive differ:"
                               << " Truncated " << truncCN.size() << ":" << truncSensN.size() << " Extended "
                               << extenCN.size() << ":" << extenSensN.size() << " Corners " << cornrCN.size() << ":"
                               << cornrSensN.size();
    throw cms::Exception("DDException") << "The number of cells & sensitive "
                                        << "differ: Truncated " << truncCN.size() << ":" << truncSensN.size()
                                        << " Extended " << extenCN.size() << ":" << extenSensN.size() << " Corners "
                                        << cornrCN.size() << ":" << cornrSensN.size();
  }
  if ((truncCN.size() < 3) || (extenCN.size() < 3) || (cornrCN.size() < 6)) {
    edm::LogError("HGCalGeom") << "DDHGCalCell: The number of cells does not"
                               << " match with Standard: Truncated " << truncCN.size() << " < 3 Extended "
                               << extenCN.size() << " < 3"
                               << " Corners " << cornrCN.size() << " < 6";
    throw cms::Exception("DDException") << "Wrong size of truncated|extended"
                                        << "|corner cells: " << truncCN.size() << ":" << extenCN.size() << ":"
                                        << cornrCN.size();
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: Wafer r " << cms::convert2mm(waferSize) << " T "
                                << cms::convert2mm(waferT) << " Cell T " << cms::convert2mm(cellT) << " Cells/Wafer "
                                << nCells << " Material " << material << "Sensitive Position " << posSens
                                << " NameSpace " << ns.name() << " Full Cell: " << fullCN << ":" << fullSensN
                                << " Separation " << waferSeparation << ":" << addWaferSeparation;
  for (unsigned int k = 0; k < truncCN.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: Truncated Cell[" << k << "] " << truncCN[k] << ":" << truncSensN[k];
  for (unsigned int k = 0; k < extenCN.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: Extended Cell[" << k << "] " << extenCN[k] << ":" << extenSensN[k];
  for (unsigned int k = 0; k < cornrCN.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: Corner Cell[" << k << "] " << cornrCN[k] << ":" << cornrSensN[k];
#endif

  dd4hep::Material matter = ns.material(material);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << material << " initialized";
#endif
  dd4hep::Volume glog1, glog2;

  static const double sqrt3 = std::sqrt(3.0);
  static const double waf2cell = 3.0;
  static const double cornerfac = 2.5;
  double R =
      (addWaferSeparation <= 1) ? waferSize / (waf2cell * nCells) : (waferSize + waferSeparation) / (waf2cell * nCells);
  double r = 0.5 * R * sqrt3;
  double dx1 = R;
  double dx2 = 0.5 * dx1;
  double dx3 = cornerfac * dx2;
  double dx4 = 0.5 * dx2;
  double dx5 = (addWaferSeparation == 0) ? 0.0 : waferSeparation * 0.5;
  double dx6 = dx5 * 0.5;
  double dx7 = dx5;
  double dy1 = r;
  double dy2 = 0.5 * dy1;
  double dy3 = 1.5 * dy1;
  double dy4 = dx5 * 0.5 * sqrt3;
  double dy5 = dx5 * 2 / sqrt3;
  double dy6 = dy5 * 0.5;
  std::vector<double> xx = {
      dx1, dx2, -dx2, -dx1, -dx2, dx2, dx3, dx1, dx4, -dx4, -dx1, -dx3, -dx3, -dx1, -dx4, dx4, dx1, dx3};
  std::vector<double> yy = {
      0, dy1, dy1, 0, -dy1, -dy1, dy2, dy1, dy3, dy3, dy1, dy2, -dy2, -dy1, -dy3, -dy3, -dy1, -dy2};
  std::vector<double> txx = {dx5, dx6, -dx6, -dx5, -dx6, dx6, dx7, 0, -dx7, -dx7, 0, dx7};
  std::vector<double> tyy = {0, dy4, dy4, 0, -dy4, -dy4, dy6, dy5, dy6, -dy6, -dy5, -dy6};
  double zpos = (posSens == 0) ? -0.5 * (waferT - cellT) : 0.5 * (waferT - cellT);
  dd4hep::Position tran(0, 0, zpos);

  // First the full cell
  std::vector<double> xw = {xx[0], xx[1], xx[2], xx[3], xx[4], xx[5]};
  std::vector<double> yw = {yy[0], yy[1], yy[2], yy[3], yy[4], yy[5]};
  std::vector<double> zw = {-0.5 * waferT, 0.5 * waferT};
  std::vector<double> zx(2, 0), zy(2, 0), scale(2, 1.0);
  dd4hep::Solid solid = dd4hep::ExtrudedPolygon(xw, yw, zw, zx, zy, scale);
  ns.addSolidNS(ns.prepend(fullCN), solid);
  glog1 = dd4hep::Volume(solid.name(), solid, matter);
  ns.addVolumeNS(glog1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() << " extruded polygon made of " << material
                                << " z|x|y|s (0) " << cms::convert2mm(zw[0]) << ":" << cms::convert2mm(zx[0]) << ":"
                                << cms::convert2mm(zy[0]) << ":" << scale[0] << " z|x|y|s (1) "
                                << cms::convert2mm(zw[1]) << ":" << cms::convert2mm(zx[1]) << ":"
                                << cms::convert2mm(zy[1]) << ":" << scale[1] << " and " << xw.size() << " edges";
  for (unsigned int k = 0; k < xw.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << cms::convert2mm(xw[k]) << ":" << cms::convert2mm(yw[k]);
#endif
  std::vector<double> zc = {-0.5 * cellT, 0.5 * cellT};
  solid = dd4hep::ExtrudedPolygon(xw, yw, zc, zx, zy, scale);
  ns.addSolidNS(ns.prepend(fullSensN), solid);
  glog2 = dd4hep::Volume(solid.name(), solid, matter);
  ns.addVolumeNS(glog2);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() << " extruded polygon made of " << material
                                << " z|x|y|s (0) " << cms::convert2mm(zc[0]) << ":" << cms::convert2mm(zx[0]) << ":"
                                << cms::convert2mm(zy[0]) << ":" << scale[0] << " z|x|y|s (1) "
                                << cms::convert2mm(zc[1]) << ":" << cms::convert2mm(zx[1]) << ":"
                                << cms::convert2mm(zy[1]) << ":" << scale[1] << " and " << xw.size() << " edges";
  for (unsigned int k = 0; k < xw.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << cms::convert2mm(xw[k]) << ":" << cms::convert2mm(yw[k]);
#endif

  glog1.placeVolume(glog2, 1, tran);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << glog2.name() << " number 1 position in " << glog1.name()
                                << " at (0,0," << cms::convert2mm(zpos) << ") with no rotation";
#endif

  static constexpr int ir0[] = {0, 2, 4, 1, 3, 5};
  static constexpr int ir1[] = {1, 3, 5, 2, 4, 0};
  static constexpr int ir2[] = {2, 4, 0, 3, 5, 1};
  static constexpr int ir3[] = {3, 5, 1, 4, 0, 2};
  static constexpr int ir4[] = {5, 1, 3, 0, 2, 4};
  static constexpr int tr[] = {1, 3, 5, 2, 4, 0};
  for (unsigned int i = 0; i < truncCN.size(); ++i) {
    std::vector<double> xw = {xx[ir0[i]], xx[ir1[i]], xx[ir2[i]], xx[ir3[i]] + txx[tr[i]], xx[ir4[i]] + txx[tr[i]]};
    std::vector<double> yw = {yy[ir0[i]], yy[ir1[i]], yy[ir2[i]], yy[ir3[i]] + tyy[tr[i]], yy[ir4[i]] + tyy[tr[i]]};

    solid = dd4hep::ExtrudedPolygon(xw, yw, zw, zx, zy, scale);
    ns.addSolidNS(ns.prepend(truncCN[i]), solid);
    glog1 = dd4hep::Volume(solid.name(), solid, matter);
    ns.addVolumeNS(glog1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() << " extruded polygon made of " << material
                                  << " z|x|y|s (0) " << cms::convert2mm(zw[0]) << ":" << cms::convert2mm(zx[0]) << ":"
                                  << cms::convert2mm(zy[0]) << ":" << scale[0] << " z|x|y|s (1) "
                                  << cms::convert2mm(zw[1]) << ":" << cms::convert2mm(zx[1]) << ":"
                                  << cms::convert2mm(zy[1]) << ":" << scale[1] << " and " << xw.size() << " edges";
    for (unsigned int k = 0; k < xw.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << cms::convert2mm(xw[k]) << ":" << cms::convert2mm(yw[k]);
#endif

    solid = dd4hep::ExtrudedPolygon(xw, yw, zc, zx, zy, scale);
    ns.addSolidNS(ns.prepend(truncSensN[i]), solid);
    glog2 = dd4hep::Volume(solid.name(), solid, matter);
    ns.addVolumeNS(glog2);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() << " extruded polygon made of " << material
                                  << " z|x|y|s (0) " << cms::convert2mm(zc[0]) << ":" << cms::convert2mm(zx[0]) << ":"
                                  << cms::convert2mm(zy[0]) << ":" << scale[0] << " z|x|y|s (1) "
                                  << cms::convert2mm(zc[1]) << ":" << cms::convert2mm(zx[1]) << ":"
                                  << cms::convert2mm(zy[1]) << ":" << scale[1] << " and " << xw.size() << " edges";
    for (unsigned int k = 0; k < xw.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << cms::convert2mm(xw[k]) << ":" << cms::convert2mm(yw[k]);
#endif
    glog1.placeVolume(glog2, 1, tran);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << glog2.name() << " number 1 position in " << glog1.name()
                                  << " at (0,0," << cms::convert2mm(zpos) << ") with no rotation";
#endif
  }

  static constexpr int ie0[] = {1, 3, 5, 0, 2, 4};
  static constexpr int ie1[] = {2, 4, 0, 1, 3, 5};
  static constexpr int ie2[] = {3, 5, 1, 2, 4, 0};
  static constexpr int ie3[] = {14, 6, 10, 12, 16, 8};
  static constexpr int ie4[] = {17, 9, 13, 15, 7, 11};
  static constexpr int te[] = {2, 4, 0, 1, 3, 5};
  for (unsigned int i = 0; i < extenCN.size(); ++i) {
    std::vector<double> xw = {xx[ie0[i]], xx[ie1[i]], xx[ie2[i]], xx[ie3[i]] + txx[te[i]], xx[ie4[i]] + txx[te[i]]};
    std::vector<double> yw = {yy[ie0[i]], yy[ie1[i]], yy[ie2[i]], yy[ie3[i]] + tyy[te[i]], yy[ie4[i]] + tyy[te[i]]};
    solid = dd4hep::ExtrudedPolygon(xw, yw, zw, zx, zy, scale);
    ns.addSolidNS(ns.prepend(extenCN[i]), solid);
    glog1 = dd4hep::Volume(solid.name(), solid, matter);
    ns.addVolumeNS(glog1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() << " extruded polygon made of " << material
                                  << " z|x|y|s (0) " << cms::convert2mm(zw[0]) << ":" << cms::convert2mm(zx[0]) << ":"
                                  << cms::convert2mm(zy[0]) << ":" << scale[0] << " z|x|y|s (1) "
                                  << cms::convert2mm(zw[1]) << ":" << cms::convert2mm(zx[1]) << ":"
                                  << cms::convert2mm(zy[1]) << ":" << scale[1] << " and " << xw.size() << " edges";
    for (unsigned int k = 0; k < xw.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << cms::convert2mm(xw[k]) << ":" << cms::convert2mm(yw[k]);
#endif
    solid = dd4hep::ExtrudedPolygon(xw, yw, zc, zx, zy, scale);
    ns.addSolidNS(ns.prepend(extenSensN[i]), solid);
    glog2 = dd4hep::Volume(solid.name(), solid, matter);
    ns.addVolumeNS(glog2);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() << " extruded polygon made of " << material
                                  << " z|x|y|s (0) " << cms::convert2mm(zc[0]) << ":" << cms::convert2mm(zx[0]) << ":"
                                  << cms::convert2mm(zy[0]) << ":" << scale[0] << " z|x|y|s (1) "
                                  << cms::convert2mm(zc[1]) << ":" << cms::convert2mm(zx[1]) << ":"
                                  << cms::convert2mm(zy[1]) << ":" << scale[1] << " and " << xw.size() << " edges";
    for (unsigned int k = 0; k < xw.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << cms::convert2mm(xw[k]) << ":" << cms::convert2mm(yw[k]);
#endif
    glog1.placeVolume(glog2, 1, tran);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << glog2.name() << " number 1 position in " << glog1.name()
                                  << " at (0,0," << cms::convert2mm(zpos) << ") with no rotation";
#endif
  }

  static constexpr int ic0[] = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
  static constexpr int ic1[] = {1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0};
  static constexpr int ic2[] = {10, 3, 14, 5, 6, 1, 2, 12, 4, 16, 0, 8};
  static constexpr int ic3[] = {3, 5, 5, 1, 1, 3, 4, 4, 0, 0, 2, 2};
  static constexpr int ic4[] = {5, 17, 1, 9, 3, 13, 15, 0, 7, 2, 11, 4};
  static constexpr int tc[] = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5};
  for (unsigned int i = 0; i < cornrCN.size(); ++i) {
    std::vector<double> xw = {
        xx[ic0[i]], xx[ic1[i]], xx[ic2[i]] + txx[tc[i]], xx[ic3[i]] + txx[tc[i] + 6], xx[ic4[i]] + txx[(tc[i] + 1) % 6]};
    std::vector<double> yw = {
        yy[ic0[i]], yy[ic1[i]], yy[ic2[i]] + tyy[tc[i]], yy[ic3[i]] + tyy[tc[i] + 6], yy[ic4[i]] + tyy[(tc[i] + 1) % 6]};
    solid = dd4hep::ExtrudedPolygon(xw, yw, zw, zx, zy, scale);
    ns.addSolidNS(ns.prepend(cornrCN[i]), solid);
    glog1 = dd4hep::Volume(solid.name(), solid, matter);
    ns.addVolumeNS(glog1);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() << " extruded polygon made of " << material
                                  << " z|x|y|s (0) " << cms::convert2mm(zw[0]) << ":" << cms::convert2mm(zx[0]) << ":"
                                  << cms::convert2mm(zy[0]) << ":" << scale[0] << " z|x|y|s (1) "
                                  << cms::convert2mm(zw[1]) << ":" << cms::convert2mm(zx[1]) << ":"
                                  << cms::convert2mm(zy[1]) << ":" << scale[1] << " and " << xw.size() << " edges";
    for (unsigned int k = 0; k < xw.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << cms::convert2mm(xw[k]) << ":" << cms::convert2mm(yw[k]);
#endif
    solid = dd4hep::ExtrudedPolygon(xw, yw, zc, zx, zy, scale);
    ns.addSolidNS(ns.prepend(cornrSensN[i]), solid);
    glog2 = dd4hep::Volume(solid.name(), solid, matter);
    ns.addVolumeNS(glog2);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << solid.name() << " extruded polygon made of " << material
                                  << " z|x|y|s (0) " << cms::convert2mm(zc[0]) << ":" << cms::convert2mm(zx[0]) << ":"
                                  << cms::convert2mm(zy[0]) << ":" << scale[0] << " z|x|y|s (1) "
                                  << cms::convert2mm(zc[1]) << ":" << cms::convert2mm(zx[1]) << ":"
                                  << cms::convert2mm(zy[1]) << ":" << scale[1] << " and " << xw.size() << " edges";
    for (unsigned int k = 0; k < xw.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << cms::convert2mm(xw[k]) << ":" << cms::convert2mm(yw[k]);
#endif
    glog1.placeVolume(glog2, 1, tran);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "DDHGCalCell: " << glog2.name() << " number 1 position in " << glog1.name()
                                  << " at (0,0," << cms::convert2mm(zpos) << ") with no rotation";
#endif
  }
  return cms::s_executed;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalCell, algorithm)
