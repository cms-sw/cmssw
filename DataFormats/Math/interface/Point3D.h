#ifndef Math_Point3D_h
#define Math_Point3D_h
// $Id: Point3D.h,v 1.3 2005/12/15 17:42:10 llista Exp $
#include <Rtypes.h>
#include <Math/Cartesian3D.h>
#include <Math/Point3D.h>

namespace math {
  /// point in space with cartesian internal representation
  typedef ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Double32_t> > XYZPoint;
}

#endif
