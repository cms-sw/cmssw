#ifndef Math_Vector3D_h
#define Math_Vector3D_h
// $Id: Vector3D.h,v 1.11 2006/09/22 08:12:04 llista Exp $
#include "Rtypes.h"
#include "Math/Cartesian3D.h"
#include "Math/Polar3D.h"
#include "Math/CylindricalEta3D.h"
#include "Math/Vector3D.h"

namespace math {

  /// spatial vector with cartesian internal representation
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> > XYZVectorD;
  /// spatial vector with cylindrical internal representation using pseudorapidity
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<double> > RhoEtaPhiVectorD;
  /// spatial vector with polar internal representation
  /// WARNING: ROOT dictionary not provided for the type below
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<double> > RThetaPhiVectorD;


  /// spatial vector with cartesian internal representation
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float> > XYZVectorF;
  /// spatial vector with cylindrical internal representation using pseudorapidity
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<float> > RhoEtaPhiVectorF;
  /// spatial vector with polar internal representation
  /// WARNING: ROOT dictionary not provided for the type below
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<float> > RThetaPhiVectorF;

  /// vector in local coordinate system
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float>, ROOT::Math::LocalCoordinateSystemTag> LocalVector;
  /// vector in glovbal coordinate system
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float>, ROOT::Math::GlobalCoordinateSystemTag> GlobalVector;

  /// spatial vector with cartesian internal representation
  typedef XYZVectorD XYZVector;
  /// spatial vector with cylindrical internal representation using pseudorapidity
  typedef RhoEtaPhiVectorD RhoEtaPhiVector;
  /// spatial vector with polar internal representation
  typedef RThetaPhiVectorD RThetaPhiVector;
}

#endif
