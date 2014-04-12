// #include "Utilities/Configuration/interface/Architecture.h"

#include "MagneticField/VolumeGeometry/interface/MagVolume.h"
#include "MagneticField/VolumeGeometry/interface/MagneticFieldProvider.h"

MagVolume::~MagVolume() {
  if (theProviderOwned) delete theProvider;
}


MagVolume::LocalVector MagVolume::fieldInTesla( const LocalPoint& lp) const 
{
  return theProvider->valueInTesla(lp)*theScalingFactor;
}

MagVolume::GlobalVector MagVolume::fieldInTesla( const GlobalPoint& gp) const
{
  return toGlobal( theProvider->valueInTesla( toLocal(gp)))*theScalingFactor;
}

