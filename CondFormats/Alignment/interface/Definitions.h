#ifndef CondFormats_Alignment_Definitions_H
#define CondFormats_Alignment_Definitions_H

/** \namespace align
 *
 *  Namespace for common type definitions used in alignment.
 *
 *  $Date: 2007/04/09 00:26:08 $
 *  $Revision: 1.2 $
 *  \author Chung Khim Lae
 */

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/GeometrySurface/interface/TkRotation.h"
#include "DataFormats/GeometryVector/interface/GlobalTag.h"
#include "DataFormats/GeometryVector/interface/LocalTag.h"
#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "DataFormats/GeometryVector/interface/Vector3DBase.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Vector.h"
#include <cstdint>

namespace align {
  typedef uint32_t ID;
  typedef double Scalar;

  typedef TkRotation<Scalar> RotationType;
  typedef Point3DBase<Scalar, GlobalTag> PositionType;
  typedef Point3DBase<Scalar, GlobalTag> GlobalPoint;
  typedef Point3DBase<Scalar, LocalTag> LocalPoint;
  typedef Vector3DBase<Scalar, GlobalTag> GlobalVector;
  typedef Vector3DBase<Scalar, LocalTag> LocalVector;

  typedef AlgebraicVector EulerAngles;
  typedef AlgebraicMatrix Derivatives;
  typedef math::Vector<6>::type AlignParams;
  typedef math::Error<6>::type ErrorMatrix;
}  // namespace align

#endif
