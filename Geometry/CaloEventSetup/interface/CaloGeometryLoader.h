#ifndef GEOMETRY_ECALGEOMETRYLOADER_H
#define GEOMETRY_ECALGEOMETRYLOADER_H

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"

#include "DD4hep/DD4hepUnits.h"
#include "CLHEP/Geometry/Transform3D.h"
#include <string>
#include <vector>

/** \class CaloGeometryLoader<T>
 *
 * Templated class for calo subdetector geometry loaders either from DDD or DD4hep.
*/

template <class T>
class CaloGeometryLoader {
public:
  using ParmVec = std::vector<double>;
  using PtrType = std::unique_ptr<CaloSubdetectorGeometry>;
  using ParVec = CaloSubdetectorGeometry::ParVec;
  using ParVecVec = CaloSubdetectorGeometry::ParVecVec;

  static constexpr double k_ScaleFromDDD = 0.1;
  static constexpr double k_ScaleFromDD4hep = (1.0 / dd4hep::cm);

  virtual ~CaloGeometryLoader() = default;

  CaloGeometryLoader() = default;

  PtrType load(const DDCompactView* cpv, const Alignments* alignments = nullptr, const Alignments* globals = nullptr);
  PtrType load(const cms::DDCompactView* cpv,
               const Alignments* alignments = nullptr,
               const Alignments* globals = nullptr);

private:
  void makeGeometry(const DDCompactView* cpv, T* geom, const Alignments* alignments, const Alignments* globals);
  void makeGeometry(const cms::DDCompactView* cpv, T* geom, const Alignments* alignments, const Alignments* globals);

  void fillNamedParams(const DDFilteredView& fv, T* geom);
  void fillNamedParams(const cms::DDFilteredView& fv, T* geom);

  void fillGeom(T* geom, const ParmVec& pv, const HepGeom::Transform3D& tr, const DetId& id, const double& scale);

  unsigned int getDetIdForDDDNode(const DDFilteredView& fv);
  unsigned int getDetIdForDD4hepNode(const cms::DDFilteredView& fv);

  typename T::NumberingScheme m_scheme;
};

#endif
