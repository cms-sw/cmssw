#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameWrapper.h"

PixelBarrelNameWrapper::PixelBarrelNameWrapper(const edm::ParameterSet& iConfig, const DetId & detId) :
  isUpgrade(iConfig.getUntrackedParameter<bool>("isUpgrade",false))
{
  if (!isUpgrade)
    pixelBarrelNameBase = new PixelBarrelName(detId);
  else
    pixelBarrelNameBase = new PixelBarrelNameUpgrade(detId);
}
