#include "MagneticField/Interpolation/interface/MFGrid3D.h"
#include "MagneticField/VolumeGeometry/interface/MagVolumeOutsideValidity.h"
#include "MagneticField/VolumeGeometry/interface/MagExceptions.h"

MFGrid::LocalVector MFGrid3D::valueInTesla( const LocalPoint& p) const
{
  try {
    return uncheckedValueInTesla( p);
  }
  catch ( GridInterpolator3DException& outside) {
    LocalPoint lower = fromGridFrame( outside.a1_, outside.b1_, outside.c1_);
    LocalPoint upper = fromGridFrame( outside.a2_, outside.b2_, outside.c2_);
    throw MagVolumeOutsideValidity( lower, upper);
  }

}
