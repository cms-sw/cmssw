#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalGeometry/interface/CaloGeometryDBHGCal.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryDBEP.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryDBReader.h"

template <>
CaloGeometryDBEP<HGCalGeometry, CaloGeometryDBReader>::PtrType
CaloGeometryDBEP<HGCalGeometry, CaloGeometryDBReader>::produceAligned(
    const typename HGCalGeometry::AlignedRecord& iRecord) {
  TrVec tvec;
  DimVec dvec;
  IVec ivec;
  IVec dins;

  edm::LogVerbatim("HGCalGeom") << "Reading HGCalGeometry " << calogeometryDBEPimpl::nameHGCal;
  const auto& pG = iRecord.get(geometryToken_);

  tvec = pG.getTranslation();
  dvec = pG.getDimension();
  ivec = pG.getIndexes();
  dins = pG.getDenseIndices();
  //*********************************************************************************************
  const auto& topology = iRecord.get(additionalTokens_.topology);

  assert(dvec.size() <= topology.totalGeomModules() * HGCalGeometry::k_NumberOfParametersPerShape);
  HGCalGeometry* hcg = new HGCalGeometry(topology);
  PtrType ptr(hcg);

  ptr->allocateCorners(topology.ncells());
  ptr->allocatePar(HGCalGeometry::k_NumberOfShapes, HGCalGeometry::k_NumberOfParametersPerShape);

  const unsigned int nTrParm(ptr->numberOfTransformParms());
  const unsigned int nPerShape(HGCalGeometry::k_NumberOfParametersPerShape);

  for (auto it : dins) {
    DetId id = topology.encode(topology.geomDenseId2decId(it));
    // get layer
    int layer = ivec[it];

    // get transformation
    const unsigned int jj(it * nTrParm);
    Tr3D tr;
    const ROOT::Math::Translation3D tl(tvec[jj], tvec[jj + 1], tvec[jj + 2]);
    const ROOT::Math::EulerAngles ea(6 == nTrParm ? ROOT::Math::EulerAngles(tvec[jj + 3], tvec[jj + 4], tvec[jj + 5])
                                                  : ROOT::Math::EulerAngles());
    const ROOT::Math::Transform3D rt(ea, tl);
    double xx, xy, xz, dx, yx, yy, yz, dy, zx, zy, zz, dz;
    rt.GetComponents(xx, xy, xz, dx, yx, yy, yz, dy, zx, zy, zz, dz);
    tr = Tr3D(CLHEP::HepRep3x3(xx, xy, xz, yx, yy, yz, zx, zy, zz), CLHEP::Hep3Vector(dx, dy, dz));

    // get parameters
    DimVec dims;
    dims.reserve(nPerShape);

    DimVec::const_iterator dsrc(dvec.begin() + layer * nPerShape);
    for (unsigned int j(0); j != nPerShape; ++j) {
      dims.emplace_back(*dsrc);
      ++dsrc;
    }

    std::vector<GlobalPoint> corners(FlatHexagon::ncorner_);

    FlatHexagon::createCorners(dims, tr, corners);

    const CCGFloat* myParm(CaloCellGeometry::getParmPtr(dims, ptr->parMgr(), ptr->parVecVec()));
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

    ptr->newCell(front, back, corners[0], myParm, id);
  }

  ptr->initializeParms();  // initializations; must happen after cells filled

  return ptr;
}

template class CaloGeometryDBEP<HGCalGeometry, CaloGeometryDBReader>;

typedef CaloGeometryDBEP<HGCalGeometry, CaloGeometryDBReader> HGCalGeometryFromDBEP;

DEFINE_FWK_EVENTSETUP_MODULE(HGCalGeometryFromDBEP);
