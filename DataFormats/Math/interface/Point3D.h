#ifndef Math_Point3D_h
#define Math_Point3D_h
// $Id: Point3D.h,v 1.4 2006/03/06 12:45:29 llista Exp $
#include <Rtypes.h>
#include <Math/Cartesian3D.h>
#include <Math/Point3D.h>

namespace math {
  /// point in space with cartesian internal representation
  typedef ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Double32_t> > XYZPoint;

  /// point in space with cartesian internal representation
  typedef ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> > XYZPointD;

  /// point in space with cartesian internal representation
  typedef ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float> > XYZPointF;
}

#endif
