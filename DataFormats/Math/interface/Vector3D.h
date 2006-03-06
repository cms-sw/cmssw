#ifndef Math_Vector3D_h
#define Math_Vector3D_h
// $Id: Vector3D.h,v 1.4 2006/01/30 21:31:11 llista Exp $
#include <Rtypes.h>
#include <Math/Cartesian3D.h>
#include <Math/Polar3D.h>
#include <Math/CylindricalEta3D.h>
#include <Math/Vector3D.h>

namespace math {
  /// spatial vector with cartesian internal representation
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Double32_t> > XYZVector;
  /// spatial vector with cylindrical internal representation using pseudorapidity
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<Double32_t> > RhoEtaPhiVector;
  /// spatial vector with polar internal representation
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<Double32_t> > RThetaPhiVector;
}

#endif
