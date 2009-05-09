#include "CondFormats/SiPixelObjects/interface/PixelCaenAliasBarrel.h"
#include "CondFormats/SiPixelObjects/interface/PixelCaenAliasEndcap.h"
#include "CondFormats/SiPixelObjects/interface/PixelDCSObject.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/SiPixelObjects/interface/PixelCaenChannels.h"

PixelCaenChannels::PixelCaenChannels(const PixelDCSObject<CaenChannel>& channels,
                                     const DetId& id)
{
  PixelCaenAlias* alias = 0;

  switch ( id.subdetId() )
  {
    case PixelSubdetector::PixelBarrel: alias = new PixelCaenAliasBarrel(id); break;
    case PixelSubdetector::PixelEndcap: alias = new PixelCaenAliasEndcap(id); break;

    default: throw cms::Exception("PixelCaenChannels") << "Invalid detid " << id;
  }

  theDigitalLV = &channels.getValue( alias->digitalLV() );
  theAnalogLV = &channels.getValue( alias->analogLV() );
  theBiasedHV = &channels.getValue( alias->biasedHV() );

  delete alias;
}
