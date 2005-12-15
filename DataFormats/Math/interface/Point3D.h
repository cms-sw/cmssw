#ifndef Math_Point3D_h
#define Math_Point3D_h
// $Id: Point3D.h,v 1.1 2005/12/15 01:59:50 llista Exp $
#include <Rtypes.h>
#include <Math/Cartesian3D.h>
#include <Math/Point3D.h>

namespace math {
  typedef ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Double32_t> > Point3D;
}

#endif
