#include "MFGrid3D.h"
#include "MagneticField/VolumeGeometry/interface/MagVolumeOutsideValidity.h"
#include "MagneticField/VolumeGeometry/interface/MagExceptions.h"

MFGrid::LocalVector MFGrid3D::valueInTesla( const LocalPoint& p) const
{
  try {
    return uncheckedValueInTesla( p);
  }
  catch ( GridInterpolator3DException& outside) {
    double *limits = outside.limits();
    LocalPoint lower = fromGridFrame( limits[0], limits[1], limits[2]);
    LocalPoint upper = fromGridFrame( limits[3], limits[4], limits[5]);
    throw MagVolumeOutsideValidity( lower, upper);
  }

}
