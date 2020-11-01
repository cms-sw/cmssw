#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloGeometryLoader.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"

typedef CaloGeometryLoader<EcalEndcapGeometry> EcalEGL;

template <>
void EcalEGL::fillGeom(EcalEndcapGeometry* geom,
                       const EcalEGL::ParmVec& vv,
                       const HepGeom::Transform3D& tr,
                       const DetId& id,
                       const double& scale);
template <>
void EcalEGL::fillNamedParams(const DDFilteredView& fv, EcalEndcapGeometry* geom);
template <>
void EcalEGL::fillNamedParams(const cms::DDFilteredView& fv, EcalEndcapGeometry* geom);

#include "Geometry/CaloEventSetup/interface/CaloGeometryLoader.icc"

template class CaloGeometryLoader<EcalEndcapGeometry>;
typedef CaloCellGeometry::CCGFloat CCGFloat;

template <>
void EcalEGL::fillGeom(EcalEndcapGeometry* geom,
                       const EcalEGL::ParmVec& vv,
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
    const CCGFloat factor(1 == i || 2 == i || 6 == i || 10 == i ? 1 : scale);
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
void EcalEGL::fillNamedParams(const DDFilteredView& _fv, EcalEndcapGeometry* geom) {
  DDFilteredView fv = _fv;
  bool doSubDets = fv.firstChild();
  while (doSubDets) {
    DDsvalues_type sv(fv.mergedSpecifics());

    //ncrys
    DDValue valNcrys("ncrys");
    if (DDfetch(&sv, valNcrys)) {
      const std::vector<double>& fvec = valNcrys.doubles();

      // this parameter can only appear once
      assert(fvec.size() == 1);
      geom->setNumberOfCrystalPerModule((int)fvec[0]);
    } else
      continue;

    //nmods
    DDValue valNmods("nmods");
    if (DDfetch(&sv, valNmods)) {
      const std::vector<double>& fmvec = valNmods.doubles();

      // there can only be one such value
      assert(fmvec.size() == 1);
      geom->setNumberOfModules((int)fmvec[0]);
    }

    break;

    doSubDets = fv.nextSibling();  // go to next layer
  }
}

template <>
void EcalEGL::fillNamedParams(const cms::DDFilteredView& fv, EcalEndcapGeometry* geom) {
  const std::string specName = "ecal_ee";

  //ncrys
  std::vector<double> tempD = fv.get<std::vector<double> >(specName, "ncrys");
  assert(tempD.size() == 1);
  geom->setNumberOfCrystalPerModule((int)tempD[0]);

  //nmods
  tempD = fv.get<std::vector<double> >(specName, "nmods");
  assert(tempD.size() == 1);
  geom->setNumberOfModules((int)tempD[0]);
}
