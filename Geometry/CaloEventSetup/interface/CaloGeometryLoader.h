#ifndef GEOMETRY_ECALGEOMETRYLOADER_H
#define GEOMETRY_ECALGEOMETRYLOADER_H

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "CondFormats/Alignment/interface/Alignments.h"

#include "CLHEP/Geometry/Transform3D.h"
#include <string>
#include <vector>

/** \class CaloGeometryLoader<T, D>
 *
 * Templated class for calo subdetector geometry loaders either from DDD or DD4hep.
*/

template <class T, class D>
class CaloGeometryLoader {
public:
  using ParmVec = std::vector<double>;
  using PtrType = std::unique_ptr<CaloSubdetectorGeometry>;
  using ParVec = CaloSubdetectorGeometry::ParVec;
  using ParVecVec = CaloSubdetectorGeometry::ParVecVec;

  static const double k_ScaleFromDDDtoGeant;

  CaloGeometryLoader<T, D>();

  virtual ~CaloGeometryLoader<T, D>() {}

  PtrType load(const D* cpv, const Alignments* alignments = nullptr, const Alignments* globals = nullptr);

private:
  void makeGeometry(const D* cpv, T* geom, const Alignments* alignments, const Alignments* globals);

  template <class F>
  void fillNamedParams(const F& fv, T* geom);

  void fillGeom(T* geom, const ParmVec& pv, const HepGeom::Transform3D& tr, const DetId& id);

  template <class F>
  unsigned int getDetIdForDDDNode(const F& fv);

  typename T::NumberingScheme m_scheme;
};

#endif
