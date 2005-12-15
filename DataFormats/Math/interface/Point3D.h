#ifndef Math_Point3D_h
#define Math_Point3D_h
// $Id: Point3D.h,v 1.2 2005/12/15 03:39:02 llista Exp $
#include <Rtypes.h>
#include <Math/Cartesian3D.h>
#include <Math/Point3D.h>

namespace math {
  typedef ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Double32_t> > XYZPoint;
}

#endif
