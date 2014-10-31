#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameWrapper.h"

PixelEndcapNameWrapper::PixelEndcapNameWrapper(const edm::ParameterSet& iConfig, const DetId & detId) :
  isUpgrade(iConfig.getUntrackedParameter<bool>("isUpgrade",false))
{
  if (!isUpgrade)
    pixelEndcapNameBase = new PixelEndcapName(detId);
  else
    pixelEndcapNameBase = new PixelEndcapNameUpgrade(detId);
}
