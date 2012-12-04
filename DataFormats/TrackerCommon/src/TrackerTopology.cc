#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

TrackerTopology::TrackerTopology( const PixelBarrelValues pxb, const PixelEndcapValues pxf,
				  const TECValues tecv, const TIBValues tibv, 
				  const TIDValues tidv, const TOBValues tobv) {
  pbVals_=pxb;
  pfVals_=pxf;
  tecVals_=tecv;
  tibVals_=tibv;
  tidVals_=tidv;
  tobVals_=tobv;
}



unsigned int TrackerTopology::layer(const DetId &id) const {
  uint32_t subdet=id.subdetId();
  if ( subdet == PixelSubdetector::PixelBarrel )
    return pxbLayer(id);
  if ( subdet == PixelSubdetector::PixelEndcap )
    return pxfDisk(id);
  if ( subdet == StripSubdetector::TIB )
    return tibLayer(id);
  if ( subdet == StripSubdetector::TID )
    return tidWheel(id);
  if ( subdet == StripSubdetector::TOB )
    return tobLayer(id);
  if ( subdet == StripSubdetector::TEC )
    return tecWheel(id);

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::layer";
  return 0;
}

unsigned int TrackerTopology::module(const DetId &id) const {
  uint32_t subdet=id.subdetId();
  if ( subdet == PixelSubdetector::PixelBarrel )
    return pxbModule(id);
  if ( subdet == PixelSubdetector::PixelEndcap )
    return pxfModule(id);
  if ( subdet == StripSubdetector::TIB )
    return tibModule(id);
  if ( subdet == StripSubdetector::TID )
    return tidModule(id);
  if ( subdet == StripSubdetector::TOB )
    return tobModule(id);
  if ( subdet == StripSubdetector::TEC )
    return tecModule(id);

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::module";
  return 0;
}

