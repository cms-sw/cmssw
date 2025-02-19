#ifndef Math_Point3D_h
#define Math_Point3D_h
// $Id: Point3D.h,v 1.9 2007/07/31 15:20:15 ratnik Exp $
#include "Math/Point3D.h"
#include "Math/GenVector/CoordinateSystemTags.h"

namespace math {
  /// point in space with cartesian internal representation
  typedef ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<double> > XYZPointD;
  /// point in space with cartesian internal representation
  typedef ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float> > XYZPointF;
  /// point in space with cartesian internal representation
  typedef XYZPointD XYZPoint;

  /// point in local coordinate system
  typedef ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float>, ROOT::Math::LocalCoordinateSystemTag> LocalPoint;
  /// point in global coordinate system
  typedef ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float>, ROOT::Math::GlobalCoordinateSystemTag> GlobalPoint;
}

#endif
