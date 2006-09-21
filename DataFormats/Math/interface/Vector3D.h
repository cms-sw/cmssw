#ifndef Math_Vector3D_h
#define Math_Vector3D_h
// $Id: Vector3D.h,v 1.8 2006/06/26 08:56:09 llista Exp $
#include <Rtypes.h>
#include <Math/Cartesian3D.h>
#include <Math/Polar3D.h>
#include <Math/CylindricalEta3D.h>
#include <Math/Vector3D.h>

namespace math {

  /// spatial vector with cartesian internal representation
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > XYZVectorD;
  /// spatial vector with cylindrical internal representation using pseudorapidity
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > RhoEtaPhiVectorD;

  /// spatial vector with cartesian internal representation
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float> > XYZVectorF;
  /// spatial vector with cylindrical internal representation using pseudorapidity
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<float> > RhoEtaPhiVectorF;

  /// vector in local coordinate system
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float>, ROOT::Math::LocalCoordinateSystemTag> LocalVector;
  /// vector in glovbal coordinate system
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float>, ROOT::Math::GlobalCoordinateSystemTag> GlobalVector;

  /// spatial vector with cartesian internal representation
  typedef XYZVectorD XYZVector;
  /// spatial vector with cylindrical internal representation using pseudorapidity
  typedef RhoEtaPhiVectorD RhoEtaPhiVector;
}

#endif
