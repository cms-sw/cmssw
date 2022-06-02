#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometryLoader.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/FlatTrd.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

//#define EDM_ML_DEBUG

typedef CaloCellGeometry::CCGFloat CCGFloat;
typedef std::vector<float> ParmVec;

HGCalGeometryLoader::HGCalGeometryLoader() : twoBysqrt3_(2.0 / std::sqrt(3.0)) {}

HGCalGeometry* HGCalGeometryLoader::build(const HGCalTopology& topology) {
  // allocate geometry
  HGCalGeometry* geom = new HGCalGeometry(topology);
  unsigned int numberOfCells = topology.totalGeomModules();  // both sides
  unsigned int numberExpected = topology.allGeomModules();
  parametersPerShape_ = (topology.tileTrapezoid() ? (int)HGCalGeometry::k_NumberOfParametersPerTrd
                                                  : (int)HGCalGeometry::k_NumberOfParametersPerHex);
  uint32_t numberOfShapes =
      (topology.tileTrapezoid() ? HGCalGeometry::k_NumberOfShapesTrd : HGCalGeometry::k_NumberOfShapes);
  HGCalGeometryMode::GeometryMode mode = topology.geomMode();
  bool test = ((mode == HGCalGeometryMode::TrapezoidModule) || (mode == HGCalGeometryMode::TrapezoidCassette));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Number of Cells " << numberOfCells << ":" << numberExpected << " for sub-detector "
                                << topology.subDetector() << " Shapes " << numberOfShapes << ":" << parametersPerShape_
                                << " mode " << mode;
#endif
  geom->allocateCorners(numberOfCells);
  geom->allocatePar(numberOfShapes, parametersPerShape_);

  // loop over modules
  ParmVec params(parametersPerShape_, 0);
  unsigned int counter(0);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalGeometryLoader with # of "
                                << "transformation matrices " << topology.dddConstants().getTrFormN() << " and "
                                << topology.dddConstants().volumes() << ":" << topology.dddConstants().sectors()
                                << " volumes";
#endif
  for (unsigned itr = 0; itr < topology.dddConstants().getTrFormN(); ++itr) {
    HGCalParameters::hgtrform mytr = topology.dddConstants().getTrForm(itr);
    int zside = mytr.zp;
    int layer = mytr.lay;
#ifdef EDM_ML_DEBUG
    unsigned int kount(0);
    edm::LogVerbatim("HGCalGeom") << "HGCalGeometryLoader:: Z:Layer " << zside << ":" << layer << " z " << mytr.h3v.z();
#endif
    if (topology.waferHexagon6()) {
      ForwardSubdetector subdet = topology.subDetector();
      for (int wafer = 0; wafer < topology.dddConstants().sectors(); ++wafer) {
        std::string code[2] = {"False", "True"};
        if (topology.dddConstants().waferInLayer(wafer, layer, true)) {
          int type = topology.dddConstants().waferTypeT(wafer);
          if (type != 1)
            type = 0;
          DetId detId = static_cast<DetId>(HGCalDetId(subdet, zside, layer, type, wafer, 0));
          const auto& w = topology.dddConstants().waferPosition(wafer, true);
          double xx = (zside > 0) ? w.first : -w.first;
          CLHEP::Hep3Vector h3v(xx, w.second, mytr.h3v.z());
          const HepGeom::Transform3D ht3d(mytr.hr, h3v);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "HGCalGeometryLoader:: Wafer:Type " << wafer << ":" << type << " DetId "
                                        << HGCalDetId(detId) << std::hex << " " << detId.rawId() << std::dec
                                        << " transf " << ht3d.getTranslation() << " and " << ht3d.getRotation();
#endif
          HGCalParameters::hgtrap vol = topology.dddConstants().getModule(wafer, true, true);
          params[FlatHexagon::k_dZ] = vol.dz;
          params[FlatHexagon::k_r] = topology.dddConstants().cellSizeHex(type);
          params[FlatHexagon::k_R] = twoBysqrt3_ * params[FlatHexagon::k_r];

          buildGeom(params, ht3d, detId, geom, 0);
          counter++;
#ifdef EDM_ML_DEBUG
          ++kount;
#endif
        }
      }
    } else if (topology.tileTrapezoid()) {
      int indx = topology.dddConstants().layerIndex(layer, true);
      int ring = topology.dddConstants().getParameter()->iradMinBH_[indx];
      int nphi = topology.dddConstants().getParameter()->scintCells(layer);
      int type = topology.dddConstants().getParameter()->scintType(layer);
      for (int md = topology.dddConstants().getParameter()->firstModule_[indx];
           md <= topology.dddConstants().getParameter()->lastModule_[indx];
           ++md) {
        for (int iphi = 1; iphi <= nphi; ++iphi) {
          HGCScintillatorDetId id(type, layer, zside * ring, iphi);
          std::pair<int, int> typm = topology.dddConstants().tileType(layer, ring, 0);
          if (typm.first >= 0) {
            id.setType(typm.first);
            id.setSiPM(typm.second);
          }
          bool ok = test ? topology.dddConstants().tileExist(zside, layer, ring, iphi) : true;
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "HGCalGeometryLoader::layer:rad:phi:type:sipm " << layer << ":"
                                        << ring * zside << ":" << iphi << ":" << type << ":" << typm.first << ":"
                                        << typm.second << " Test " << test << ":" << ok;
#endif
          if (ok) {
            DetId detId = static_cast<DetId>(id);
            const auto& w = topology.dddConstants().locateCellTrap(layer, ring, iphi, true);
            double xx = (zside > 0) ? w.first : -w.first;
            CLHEP::Hep3Vector h3v(xx, w.second, mytr.h3v.z());
            const HepGeom::Transform3D ht3d(mytr.hr, h3v);
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("HGCalGeom")
                << "HGCalGeometryLoader::rad:phi:type " << ring * zside << ":" << iphi << ":" << type << " DetId "
                << HGCScintillatorDetId(detId) << " " << std::hex << detId.rawId() << std::dec << " transf "
                << ht3d.getTranslation() << " R " << ht3d.getTranslation().perp() << " and " << ht3d.getRotation();
#endif
            HGCalParameters::hgtrap vol = topology.dddConstants().getModule(md, false, true);
            params[FlatTrd::k_dZ] = vol.dz;
            params[FlatTrd::k_Theta] = params[FlatTrd::k_Phi] = 0;
            params[FlatTrd::k_dY1] = params[FlatTrd::k_dY2] = vol.h;
            params[FlatTrd::k_dX1] = params[FlatTrd::k_dX3] = vol.bl;
            params[FlatTrd::k_dX2] = params[FlatTrd::k_dX4] = vol.tl;
            params[FlatTrd::k_Alp1] = params[FlatTrd::k_Alp2] = 0;
            params[FlatTrd::k_Cell] = topology.dddConstants().cellSizeHex(type);

            buildGeom(params, ht3d, detId, geom, 1);
            counter++;
#ifdef EDM_ML_DEBUG
            ++kount;
#endif
          }
        }
        ++ring;
      }
    } else {
      DetId::Detector det = topology.detector();
      for (int wafer = 0; wafer < topology.dddConstants().sectors(); ++wafer) {
        if (topology.dddConstants().waferInLayer(wafer, layer, true)) {
          int copy = topology.dddConstants().getParameter()->waferCopy_[wafer];
          int u = HGCalWaferIndex::waferU(copy);
          int v = HGCalWaferIndex::waferV(copy);
          int type = topology.dddConstants().getTypeHex(layer, u, v);
          DetId detId =
              (topology.isHFNose() ? static_cast<DetId>(HFNoseDetId(zside, type, layer, u, v, 0, 0))
                                   : static_cast<DetId>(HGCSiliconDetId(det, zside, type, layer, u, v, 0, 0)));
          const auto& w = topology.dddConstants().waferPosition(layer, u, v, true, true);
          double xx = (zside > 0) ? w.first : -w.first;
          CLHEP::Hep3Vector h3v(xx, w.second, mytr.h3v.z());
          const HepGeom::Transform3D ht3d(mytr.hr, h3v);
#ifdef EDM_ML_DEBUG
          if (topology.isHFNose())
            edm::LogVerbatim("HGCalGeom") << "HGCalGeometryLoader::Wafer:Type " << wafer << ":" << type << " DetId "
                                          << HFNoseDetId(detId) << std::hex << " " << detId.rawId() << std::dec
                                          << " trans " << ht3d.getTranslation() << " and " << ht3d.getRotation();
          else
            edm::LogVerbatim("HGCalGeom") << "HGCalGeometryLoader::Wafer:Type " << wafer << ":" << type << " DetId "
                                          << HGCSiliconDetId(detId) << std::hex << " " << detId.rawId() << std::dec
                                          << " trans " << ht3d.getTranslation() << " and " << ht3d.getRotation();
#endif
          HGCalParameters::hgtrap vol = topology.dddConstants().getModule(type, false, true);
          params[FlatHexagon::k_dZ] = vol.dz;
          params[FlatHexagon::k_r] = topology.dddConstants().cellSizeHex(type);
          params[FlatHexagon::k_R] = twoBysqrt3_ * params[FlatHexagon::k_r];

          buildGeom(params, ht3d, detId, geom, 0);
          counter++;
#ifdef EDM_ML_DEBUG
          ++kount;
#endif
        }
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << kount << " modules found in Layer " << layer << " Z " << zside;
#endif
  }

  geom->sortDetIds();

  if (counter != numberExpected) {
    if (topology.tileTrapezoid()) {
      edm::LogVerbatim("HGCalGeom") << "Inconsistent # of cells: expected " << numberExpected << ":" << numberOfCells
                                    << " , inited " << counter;
    } else {
      edm::LogError("HGCalGeom") << "Inconsistent # of cells: expected " << numberExpected << ":" << numberOfCells
                                 << " , inited " << counter;
      assert(counter == numberExpected);
    }
  }

  return geom;
}

void HGCalGeometryLoader::buildGeom(
    const ParmVec& params, const HepGeom::Transform3D& ht3d, const DetId& detId, HGCalGeometry* geom, int mode) {
#ifdef EDM_ML_DEBUG
  for (int i = 0; i < parametersPerShape_; ++i)
    edm::LogVerbatim("HGCalGeom") << "Parameter[" << i << "] : " << params[i];
#endif
  if (mode == 1) {
    std::vector<GlobalPoint> corners(FlatTrd::ncorner_);

    FlatTrd::createCorners(params, ht3d, corners);

    const CCGFloat* parmPtr(CaloCellGeometry::getParmPtr(params, geom->parMgr(), geom->parVecVec()));

    GlobalPoint front(0.25 * (corners[0].x() + corners[1].x() + corners[2].x() + corners[3].x()),
                      0.25 * (corners[0].y() + corners[1].y() + corners[2].y() + corners[3].y()),
                      0.25 * (corners[0].z() + corners[1].z() + corners[2].z() + corners[3].z()));

    GlobalPoint back(0.25 * (corners[4].x() + corners[5].x() + corners[6].x() + corners[7].x()),
                     0.25 * (corners[4].y() + corners[5].y() + corners[6].y() + corners[7].y()),
                     0.25 * (corners[4].z() + corners[5].z() + corners[6].z() + corners[7].z()));

    if (front.mag2() > back.mag2()) {  // front should always point to the center, so swap front and back
      std::swap(front, back);
      std::swap_ranges(corners.begin(), corners.begin() + FlatTrd::ncornerBy2_, corners.begin() + FlatTrd::ncornerBy2_);
    }
    geom->newCell(front, back, corners[0], parmPtr, detId);
  } else {
    std::vector<GlobalPoint> corners(FlatHexagon::ncorner_);

    FlatHexagon::createCorners(params, ht3d, corners);

    const CCGFloat* parmPtr(CaloCellGeometry::getParmPtr(params, geom->parMgr(), geom->parVecVec()));

    GlobalPoint front(
        FlatHexagon::oneBySix_ *
            (corners[0].x() + corners[1].x() + corners[2].x() + corners[3].x() + corners[4].x() + corners[5].x()),
        FlatHexagon::oneBySix_ *
            (corners[0].y() + corners[1].y() + corners[2].y() + corners[3].y() + corners[4].y() + corners[5].y()),
        FlatHexagon::oneBySix_ *
            (corners[0].z() + corners[1].z() + corners[2].z() + corners[3].z() + corners[4].z() + corners[5].z()));

    GlobalPoint back(
        FlatHexagon::oneBySix_ *
            (corners[6].x() + corners[7].x() + corners[8].x() + corners[9].x() + corners[10].x() + corners[11].x()),
        FlatHexagon::oneBySix_ *
            (corners[6].y() + corners[7].y() + corners[8].y() + corners[9].y() + corners[10].y() + corners[11].y()),
        FlatHexagon::oneBySix_ *
            (corners[6].z() + corners[7].z() + corners[8].z() + corners[9].z() + corners[10].z() + corners[11].z()));

    if (front.mag2() > back.mag2()) {  // front should always point to the center, so swap front and back
      std::swap(front, back);
      std::swap_ranges(
          corners.begin(), corners.begin() + FlatHexagon::ncornerBy2_, corners.begin() + FlatHexagon::ncornerBy2_);
    }
    geom->newCell(front, back, corners[0], parmPtr, detId);
  }
}
