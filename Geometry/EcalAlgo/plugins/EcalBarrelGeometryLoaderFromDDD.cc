#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryLoader.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDutils.h"

typedef CaloGeometryLoader<EcalBarrelGeometry> EcalBGL;

template <>
void EcalBGL::fillGeom(EcalBarrelGeometry* geom,
                       const EcalBGL::ParmVec& vv,
                       const HepGeom::Transform3D& tr,
                       const DetId& id,
                       const double& scale);
template <>
void EcalBGL::fillNamedParams(const DDFilteredView& fv, EcalBarrelGeometry* geom);
template <>
void EcalBGL::fillNamedParams(const cms::DDFilteredView& fv, EcalBarrelGeometry* geom);

#include "Geometry/CaloEventSetup/interface/CaloGeometryLoader.icc"

template class CaloGeometryLoader<EcalBarrelGeometry>;
typedef CaloCellGeometry::CCGFloat CCGFloat;

template <>
void EcalBGL::fillGeom(EcalBarrelGeometry* geom,
                       const EcalBGL::ParmVec& vv,
                       const HepGeom::Transform3D& tr,
                       const DetId& id,
                       const double& scale) {
  static constexpr uint32_t maxSize = 11;
  std::vector<CCGFloat> pv;
  unsigned int size = (vv.size() > maxSize) ? maxSize : vv.size();
  unsigned int ioff = (vv.size() > maxSize) ? (vv.size() - maxSize) : 0;
  pv.reserve(size);
  for (unsigned int i(0); i != size; ++i) {
    unsigned int ii = ioff + i;
    const CCGFloat factor(1 == i || 2 == i || 6 == i || 10 == i ? 1 : static_cast<CCGFloat>(scale));
    pv.emplace_back(factor * vv[ii]);
  }

  std::vector<GlobalPoint> corners(8);

  TruncatedPyramid::createCorners(pv, tr, corners);

  const CCGFloat* parmPtr(CaloCellGeometry::getParmPtr(pv, geom->parMgr(), geom->parVecVec()));

  const GlobalPoint front(0.25 * (corners[0].x() + corners[1].x() + corners[2].x() + corners[3].x()),
                          0.25 * (corners[0].y() + corners[1].y() + corners[2].y() + corners[3].y()),
                          0.25 * (corners[0].z() + corners[1].z() + corners[2].z() + corners[3].z()));

  const GlobalPoint back(0.25 * (corners[4].x() + corners[5].x() + corners[6].x() + corners[7].x()),
                         0.25 * (corners[4].y() + corners[5].y() + corners[6].y() + corners[7].y()),
                         0.25 * (corners[4].z() + corners[5].z() + corners[6].z() + corners[7].z()));

  geom->newCell(front, back, corners[0], parmPtr, id);
}

template <>
void EcalBGL::fillNamedParams(const DDFilteredView& _fv, EcalBarrelGeometry* geom) {
  DDFilteredView fv = _fv;
  bool doSubDets = fv.firstChild();

  while (doSubDets) {
    DDsvalues_type sv(fv.mergedSpecifics());

    //nxtalPhi
    DDValue valnPhi("nxtalPhi");
    if (DDfetch(&sv, valnPhi)) {
      const std::vector<double>& fvec = valnPhi.doubles();

      // this parameter can only appear once
      assert(fvec.size() == 1);
      geom->setNumXtalsPhiDirection((int)fvec[0]);
    } else
      continue;

    DDValue valnEta("nxtalEta");
    if (DDfetch(&sv, valnEta)) {
      const std::vector<double>& fmvec = valnEta.doubles();

      // there can only be one such value
      assert(fmvec.size() == 1);

      geom->setNumXtalsEtaDirection((int)fmvec[0]);
    } else
      // once we find nxtalPhi, the rest must also be defined
      assert(1 == 0);

    //EtaBaskets
    DDValue valEtaB("EtaBaskets");
    if (DDfetch(&sv, valEtaB)) {
      const std::vector<double>& ebvec = valEtaB.doubles();
      assert(!ebvec.empty());
      std::vector<int> EtaBaskets;
      EtaBaskets.resize(ebvec.size());
      for (unsigned i = 0; i < ebvec.size(); ++i)
        EtaBaskets[i] = (int)ebvec[i];
      geom->setEtaBaskets(EtaBaskets);
    } else
      // once we find nxtalPhi, the rest must also be defined
      assert(1 == 0);

    //PhiBaskets
    DDValue valPhi("PhiBaskets");
    if (DDfetch(&sv, valPhi)) {
      const std::vector<double>& pvec = valPhi.doubles();
      assert(!pvec.empty());
      geom->setBasketSizeInPhi((int)pvec[0]);
    } else
      // once we find nxtalPhi, the rest must also be defined
      assert(1 == 0);

    break;

    doSubDets = fv.nextSibling();  // go to next layer
  }
}

template <>
void EcalBGL::fillNamedParams(const cms::DDFilteredView& fv, EcalBarrelGeometry* geom) {
  const std::string specName = "ecal_eb";

  //nxtalEta
  std::vector<double> tempD = fv.get<std::vector<double> >(specName, "nxtalEta");
  assert(tempD.size() == 1);
  geom->setNumXtalsEtaDirection((int)tempD[0]);

  //nxtalPhi
  tempD = fv.get<std::vector<double> >(specName, "nxtalPhi");
  assert(tempD.size() == 1);
  geom->setNumXtalsPhiDirection((int)tempD[0]);

  //EtaBaskets
  tempD = fv.get<std::vector<double> >(specName, "EtaBaskets");
  assert(!tempD.empty());
  geom->setEtaBaskets(dbl_to_int(tempD));

  //PhiBaskets
  tempD = fv.get<std::vector<double> >(specName, "PhiBaskets");
  assert(!tempD.empty());
  geom->setBasketSizeInPhi((int)tempD[0]);
}
