#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalGeometry/interface/HGCalTBGeometryLoader.h"
#include "Geometry/HGCalGeometry/interface/HGCalTBGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/FlatTrd.h"
#include "Geometry/HGCalTBCommonData/interface/HGCalTBDDDConstants.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

//#define EDM_ML_DEBUG

typedef CaloCellGeometry::CCGFloat CCGFloat;
typedef std::vector<float> ParmVec;

HGCalTBGeometryLoader::HGCalTBGeometryLoader() : twoBysqrt3_(2.0 / std::sqrt(3.0)) {}

HGCalTBGeometry* HGCalTBGeometryLoader::build(const HGCalTBTopology& topology) {
  // allocate geometry
  HGCalTBGeometry* geom = new HGCalTBGeometry(topology);
  unsigned int numberOfCells = topology.totalGeomModules();  // both sides
  unsigned int numberExpected = topology.allGeomModules();
  parametersPerShape_ = static_cast<int>(HGCalTBGeometry::k_NumberOfParametersPerHex);
  uint32_t numberOfShapes = HGCalTBGeometry::k_NumberOfShapes;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Number of Cells " << numberOfCells << ":" << numberExpected << " for sub-detector "
                                << topology.subDetector() << " Shapes " << numberOfShapes << ":" << parametersPerShape_;
#endif
  geom->allocateCorners(numberOfCells);
  geom->allocatePar(numberOfShapes, parametersPerShape_);

  // loop over modules
  ParmVec params(parametersPerShape_, 0);
  unsigned int counter(0);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalTBGeometryLoader with # of "
                                << "transformation matrices " << topology.dddConstants().getTrFormN() << " and "
                                << topology.dddConstants().volumes() << ":" << topology.dddConstants().sectors()
                                << " volumes";
#endif
  for (unsigned itr = 0; itr < topology.dddConstants().getTrFormN(); ++itr) {
    HGCalTBParameters::hgtrform mytr = topology.dddConstants().getTrForm(itr);
    int zside = mytr.zp;
    int layer = mytr.lay;
#ifdef EDM_ML_DEBUG
    unsigned int kount(0);
    edm::LogVerbatim("HGCalGeom") << "HGCalTBGeometryLoader:: Z:Layer " << zside << ":" << layer << " z "
                                  << mytr.h3v.z();
#endif
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
        edm::LogVerbatim("HGCalGeom") << "HGCalTBGeometryLoader:: Wafer:Type " << wafer << ":" << type << " DetId "
                                      << HGCalDetId(detId) << std::hex << " " << detId.rawId() << std::dec << " transf "
                                      << ht3d.getTranslation() << " and " << ht3d.getRotation();
#endif
        HGCalTBParameters::hgtrap vol = topology.dddConstants().getModule(wafer, true, true);
        params[FlatHexagon::k_dZ] = vol.dz;
        params[FlatHexagon::k_r] = topology.dddConstants().cellSizeHex(type);
        params[FlatHexagon::k_R] = twoBysqrt3_ * params[FlatHexagon::k_r];

        buildGeom(params, ht3d, detId, geom);
        counter++;
#ifdef EDM_ML_DEBUG
        ++kount;
#endif
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << kount << " modules found in Layer " << layer << " Z " << zside;
#endif
  }

  geom->sortDetIds();

  if (counter != numberExpected) {
    edm::LogError("HGCalGeom") << "Inconsistent # of cells: expected " << numberExpected << ":" << numberOfCells
                               << " , inited " << counter;
  }

  return geom;
}

void HGCalTBGeometryLoader::buildGeom(const ParmVec& params,
                                      const HepGeom::Transform3D& ht3d,
                                      const DetId& detId,
                                      HGCalTBGeometry* geom) {
#ifdef EDM_ML_DEBUG
  for (int i = 0; i < parametersPerShape_; ++i)
    edm::LogVerbatim("HGCalGeom") << "Parameter[" << i << "] : " << params[i];
#endif
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
