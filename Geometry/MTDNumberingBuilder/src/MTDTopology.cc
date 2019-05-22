#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>
#include <limits>

MTDTopology::MTDTopology( const int& topologyMode, const BTLValues& btl, const ETLValues& etl ) 
  : mtdTopologyMode_(topologyMode),
    btlVals_(btl),
    etlVals_(etl),
    bits_per_field{
  [BTLModule] = { btlVals_.moduleStartBit_, btlVals_.moduleMask_, MTDDetId::BTL},
  [BTLTray]   = { btlVals_.trayStartBit_, btlVals_.trayMask_, MTDDetId::BTL},
  [BTLLayer]  = { btlVals_.layerStartBit_,  btlVals_.layerMask_, MTDDetId::BTL},
  [BTLSide]   = { btlVals_.sideStartBit_,  btlVals_.sideMask_, MTDDetId::BTL},
  [ETLModule] = { etlVals_.moduleStartBit_, etlVals_.moduleMask_, MTDDetId::ETL},
  [ETLRing]   = { etlVals_.ringStartBit_,  etlVals_.ringMask_,  MTDDetId::ETL},
  [ETLLayer]  = { etlVals_.layerStartBit_,  etlVals_.layerMask_,  MTDDetId::ETL},
  [ETLSide]   = { etlVals_.sideStartBit_,   etlVals_.sideMask_,   MTDDetId::ETL}
} 
{}



unsigned int MTDTopology::side(const DetId &id) const {
  uint32_t subdet=MTDDetId(id).mtdSubDetector();
  switch( subdet ) {
  case MTDDetId::BTL:    
    return btlSide(id);
  case MTDDetId::ETL:    
    return etlSide(id);
  default:
      throw cms::Exception("Invalid DetId") << "Unsupported DetId in MTDTopology::side";
  }
  return std::numeric_limits<unsigned int>::max();
}

unsigned int MTDTopology::layer(const DetId &id) const {
  uint32_t subdet=MTDDetId(id).mtdSubDetector();
  switch( subdet ) {
  case MTDDetId::BTL:    
    return btlLayer(id);
  case MTDDetId::ETL:    
    return etlLayer(id);
  default:
      throw cms::Exception("Invalid DetId") << "Unsupported DetId in MTDTopology::layer";
  }
  return std::numeric_limits<unsigned int>::max();
}

unsigned int MTDTopology::module(const DetId &id) const {
  uint32_t subdet=MTDDetId(id).mtdSubDetector();
  switch( subdet ) {
  case MTDDetId::BTL:    
    return btlModule(id);
  case MTDDetId::ETL:    
    return etlModule(id);
  default:
      throw cms::Exception("Invalid DetId") << "Unsupported DetId in MTDTopology::module";
  }
  return std::numeric_limits<unsigned int>::max();
}

unsigned int MTDTopology::tray(const DetId &id) const {
  uint32_t subdet=MTDDetId(id).mtdSubDetector();
  switch( subdet ) {
  case MTDDetId::BTL:    
    return btlTray(id);
  case MTDDetId::ETL:    
    return std::numeric_limits<unsigned int>::max();
  default:
      throw cms::Exception("Invalid DetId") << "Unsupported DetId in MTDTopology::tray";
  }
  return std::numeric_limits<unsigned int>::max();
}

unsigned int MTDTopology::ring(const DetId &id) const {
  uint32_t subdet=MTDDetId(id).mtdSubDetector();
  switch( subdet ) {
  case MTDDetId::BTL:    
    return std::numeric_limits<unsigned int>::max();
  case MTDDetId::ETL:    
    return etlModule(id);
  default:
      throw cms::Exception("Invalid DetId") << "Unsupported DetId in MTDTopology::ring";
  }
  return std::numeric_limits<unsigned int>::max();
}



std::string MTDTopology::print(DetId id) const {
  uint32_t subdet=MTDDetId(id).mtdSubDetector();
  std::stringstream strstr;

  if ( subdet == MTDDetId::BTL ) {
    unsigned int theSide   = btlSide(id);
    unsigned int theLayer  = btlLayer(id);
    unsigned int theTray   = btlTray(id);    
    unsigned int theModule = btlModule(id);
    std::string side  = (btlSide(id) == 1 ) ? "-" : "+";
    strstr << "BTL" 
	   << " Side   " << theSide << side
	   << " Layer  " << theLayer
	   << " Tray   " << theTray
           << " Module " << theModule ;
    strstr << " (" << id.rawId() << ")";
    return strstr.str();
  }

  if ( subdet == MTDDetId::ETL ) {
    unsigned int theSide   = etlSide(id);
    unsigned int theLayer  = etlLayer(id);
    unsigned int theRing   = etlRing(id);
    unsigned int theModule = etlModule(id);
    std::string side  = (etlSide(id) == 1 ) ? "-" : "+";
    strstr << "ETL" 
           << " Side   " << theSide << side
	   << " Layer  " << theLayer
	   << " Ring   " << theRing
           << " Module " << theModule ;
    strstr << " (" << id.rawId() << ")";
    return strstr.str();
  }
  throw cms::Exception("Invalid DetId") << "Unsupported DetId in MTDTopology::print";
  return strstr.str();
}



int MTDTopology::getMTDLayerNumber(const DetId &id) const {
    int layer = -1;
    uint32_t subdet=MTDDetId(id).mtdSubDetector();

    if (id.det() == DetId::Forward) {
      if (subdet == MTDDetId::BTL) {
	layer = btlLayer(id);
      } else if (id.subdetId() == MTDDetId::ETL) {
	layer = etlLayer(id);
      } else {
	edm::LogInfo("MTDTopology") << ">>> Invalid subdetId()  " ;
      }
    }
    return layer;
}

