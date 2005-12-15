#ifndef Math_Vector3D_h
#define Math_Vector3D_h
// $Id: Vector3D.h,v 1.2 2005/12/15 03:39:02 llista Exp $
#include <Rtypes.h>
#include <Math/Cartesian3D.h>
#include <Math/CylindricalEta3D.h>
#include <Math/Vector3D.h>

namespace math {
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Double32_t> > XYZVector;
  typedef ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<Double32_t> > RhoEtaPhiVector;
}

#endif
