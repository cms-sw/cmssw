#ifndef Math_Vector3D_h
#define Math_Vector3D_h
// $Id: Vector3D.h,v 1.1 2005/12/15 01:59:50 llista Exp $
#include <Rtypes.h>
#include <Math/Cartesian3D.h>
#include <Math/Vector3D.h>

namespace math {
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Double32_t> > Vector3D;
}

#endif
