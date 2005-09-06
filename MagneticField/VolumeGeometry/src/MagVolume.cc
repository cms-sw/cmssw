// #include "Utilities/Configuration/interface/Architecture.h"

#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include "MagneticField/VolumeGeometry/interface/MagneticFieldProvider.h"

MagVolume::LocalVector MagVolume::fieldInTesla( const LocalPoint& lp) const 
{
  return theProvider->valueInTesla(lp);
}

MagVolume::GlobalVector MagVolume::fieldInTesla( const GlobalPoint& gp) const
{
  return toGlobal( theProvider->valueInTesla( toLocal(gp)));
}
