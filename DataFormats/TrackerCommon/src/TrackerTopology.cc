#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include <sstream>

TrackerTopology::TrackerTopology( const PixelBarrelValues& pxb, const PixelEndcapValues& pxf,
				  const TECValues& tecv, const TIBValues& tibv, 
				  const TIDValues& tidv, const TOBValues& tobv) {
  pbVals_=pxb;
  pfVals_=pxf;
  tecVals_=tecv;
  tibVals_=tibv;
  tidVals_=tidv;
  tobVals_=tobv;
}



unsigned int TrackerTopology::side(const DetId &id) const {
  uint32_t subdet=id.subdetId();
  if ( subdet == PixelSubdetector::PixelBarrel )
    return 0;
  if ( subdet == PixelSubdetector::PixelEndcap )
    return pxfSide(id);
  if ( subdet == StripSubdetector::TIB )
    return 0;
  if ( subdet == StripSubdetector::TID )
    return tidSide(id);
  if ( subdet == StripSubdetector::TOB )
    return 0;
  if ( subdet == StripSubdetector::TEC )
    return tecSide(id);

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::side";
  return 0;
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

uint32_t TrackerTopology::Glued(const DetId &id) const {

    uint32_t subdet=id.subdetId();
    if ( subdet == PixelSubdetector::PixelBarrel )
      return 0;
    if ( subdet == PixelSubdetector::PixelEndcap )
      return 0;
    if ( subdet == StripSubdetector::TIB )
      return tibGlued(id);
    if ( subdet == StripSubdetector::TID )
      return tidGlued(id);
    if ( subdet == StripSubdetector::TOB )
      return tobGlued(id);
    if ( subdet == StripSubdetector::TEC )
      return tecGlued(id);

    throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::Glued";
    return 0;
}

uint32_t TrackerTopology::Stack(const DetId &id) const {

    uint32_t subdet=id.subdetId();
    if ( subdet == PixelSubdetector::PixelBarrel )
      return 0;
    if ( subdet == PixelSubdetector::PixelEndcap )
      return 0;
    if ( subdet == StripSubdetector::TIB )
      return tibStack(id);
    if ( subdet == StripSubdetector::TID )
      return tidStack(id);
    if ( subdet == StripSubdetector::TOB )
      return tobStack(id);
    if ( subdet == StripSubdetector::TEC )
      return tecStack(id);

    throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::Stack";
}

bool TrackerTopology::isStereo(const DetId &id) const {

    uint32_t subdet=id.subdetId();
    if ( subdet == PixelSubdetector::PixelBarrel )
      return false;
    if ( subdet == PixelSubdetector::PixelEndcap )
      return false;
    if ( subdet == StripSubdetector::TIB )
      return tibStereo(id)!=0;
    if ( subdet == StripSubdetector::TID )
      return tidStereo(id)!=0;
    if ( subdet == StripSubdetector::TOB )
      return tobStereo(id)!=0;
    if ( subdet == StripSubdetector::TEC )
      return tecStereo(id)!=0;

    throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::isStereo";
    return 0;
}

bool TrackerTopology::isRPhi(const DetId &id) const {

    uint32_t subdet=id.subdetId();
    if ( subdet == PixelSubdetector::PixelBarrel )
      return false;
    if ( subdet == PixelSubdetector::PixelEndcap )
      return false;
    if ( subdet == StripSubdetector::TIB )
      return tibRPhi(id)!=0;
    if ( subdet == StripSubdetector::TID )
      return tidRPhi(id)!=0;
    if ( subdet == StripSubdetector::TOB )
      return tobRPhi(id)!=0;
    if ( subdet == StripSubdetector::TEC )
      return tecRPhi(id)!=0;

    throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::isRPhi";
    return 0;
}
bool TrackerTopology::isLower(const DetId &id) const {

    uint32_t subdet=id.subdetId();
    if ( subdet == PixelSubdetector::PixelBarrel ) 
      return false;
    if ( subdet == PixelSubdetector::PixelEndcap )
      return false;
    if ( subdet == StripSubdetector::TIB )
      return tibLower(id)!=0;
    if ( subdet == StripSubdetector::TID )
      return tidLower(id)!=0;
    if ( subdet == StripSubdetector::TOB )
      return tobLower(id)!=0;
    if ( subdet == StripSubdetector::TEC )
      return tecLower(id)!=0;

    throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::isLower";
    return 0;

}

bool TrackerTopology::isUpper(const DetId &id) const {

    uint32_t subdet=id.subdetId();
    if ( subdet == PixelSubdetector::PixelBarrel ) 
      return false;
    if ( subdet == PixelSubdetector::PixelEndcap )
      return false;
    if ( subdet == StripSubdetector::TIB )
      return tibUpper(id)!=0;
    if ( subdet == StripSubdetector::TID )
      return tidUpper(id)!=0;
    if ( subdet == StripSubdetector::TOB )
      return tobUpper(id)!=0;
    if ( subdet == StripSubdetector::TEC )
      return tecUpper(id)!=0;

    throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::isUpper";
    return 0;
}

uint32_t TrackerTopology::PartnerDetId(const DetId &id) const {

    uint32_t subdet=id.subdetId();
    if ( subdet == PixelSubdetector::PixelBarrel )
      return 0;
    if ( subdet == PixelSubdetector::PixelEndcap )
      return 0;
    if ( subdet == StripSubdetector::TIB )
      return tibPartnerDetId(id);
    if ( subdet == StripSubdetector::TID )
      return tidPartnerDetId(id);
    if ( subdet == StripSubdetector::TOB )
      return tobPartnerDetId(id);
    if ( subdet == StripSubdetector::TEC )
      return tecPartnerDetId(id);

    throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::PartnerDetId";
    return 0;
}

std::string TrackerTopology::print(DetId id) const {
  uint32_t subdet=id.subdetId();
  std::stringstream strstr;

  if ( subdet == PixelSubdetector::PixelBarrel ) {
    strstr  << "(PixelBarrel " 
	    << pxbLayer(id) << ',' 
	    << pxbLadder(id) << ',' 
	    << pxbModule(id) << ')'; 
    return strstr.str();
  }

  if ( subdet == PixelSubdetector::PixelEndcap ) {
    strstr << "(PixelEndcap " 
	   << pxfDisk(id) << ',' 
	   << pxfBlade(id)  << ',' 
	   << pxfPanel(id)  << ',' 
	   << pxfModule(id)   << ')'; 
    return strstr.str();
  }

  if ( subdet == StripSubdetector::TIB ) {
    unsigned int              theLayer  = tibLayer(id);
    std::vector<unsigned int> theString = tibStringInfo(id);
    unsigned int              theModule = tibModule(id);
    std::string side;
    std::string part;
    side = (theString[0] == 1 ) ? "-" : "+";
    part = (theString[1] == 1 ) ? "int" : "ext";
    std::string type;
    type = (isStereo(id)) ? "stereo" : type;
    type = (isRPhi(id)) ? "r-phi" : type;
    type = (isStereo(id) || isRPhi(id)) ? type+" glued": "module";
    std::string typeUpgrade;
    typeUpgrade = (isLower(id)) ? "lower" : typeUpgrade;
    typeUpgrade = (isUpper(id)) ? "upper" : typeUpgrade;
    typeUpgrade = (isUpper(id) || isLower(id)) ? typeUpgrade+" stack": "module";
    strstr << "TIB" << side
	   << " Layer " << theLayer << " " << part
	   << " String " << theString[2];
    strstr << " Module for phase0 " << theModule << " " << type;
    strstr << " Module for phase2 " << theModule << " " << typeUpgrade;
    strstr << " (" << id.rawId() << ")";
    return strstr.str();
  }

  if ( subdet == StripSubdetector::TID ) {
    unsigned int         theDisk   = tidWheel(id);
    unsigned int         theRing   = tidRing(id);
    std::vector<unsigned int> theModule = tidModuleInfo(id);
    std::string side;
    std::string part;
    side = (tidSide(id) == 1 ) ? "-" : "+";
    part = (theModule[0] == 1 ) ? "back" : "front";
    std::string type;
    type = (isStereo(id)) ? "stereo" : type;
    type = (isRPhi(id)) ? "r-phi" : type;
    type = (isStereo(id) || isRPhi(id)) ? type+" glued": "module";
    std::string typeUpgrade;
    typeUpgrade = (isLower(id)) ? "lower" : typeUpgrade;
    typeUpgrade = (isUpper(id)) ? "upper" : typeUpgrade;
    typeUpgrade = (isUpper(id) || isLower(id)) ? typeUpgrade+" stack": "module";
    strstr << "TID" << side
	   << " Disk " << theDisk
	   << " Ring " << theRing << " " << part;
    strstr << " Module for phase0 " << theModule[1] << " " << type;
    strstr << " Module for phase2 " << theModule[1] << " " << typeUpgrade;
    strstr << " (" << id.rawId() << ")";
    return strstr.str();
  }

  if ( subdet == StripSubdetector::TOB ) {
    unsigned int              theLayer  = tobLayer(id);
    std::vector<unsigned int> theRod    = tobRodInfo(id);
    unsigned int              theModule = tobModule(id);
    std::string side;
    std::string part;
    side = (theRod[0] == 1 ) ? "-" : "+";
    std::string type;
    type = (isStereo(id)) ? "stereo" : type;
    type = (isRPhi(id)) ? "r-phi" : type;
    type = (isStereo(id) || isRPhi(id)) ? type+" glued": "module";
    std::string typeUpgrade;
    typeUpgrade = (isLower(id)) ? "lower" : typeUpgrade;
    typeUpgrade = (isUpper(id)) ? "upper" : typeUpgrade;
    typeUpgrade = (isUpper(id) || isLower(id)) ? typeUpgrade+" stack": "module";
    strstr << "TOB" << side
	   << " Layer " << theLayer
	   << " Rod " << theRod[1];
    strstr << " Module for phase0 " << theModule << " " << type;
    strstr << " Module for phase2 " << theModule << " " << typeUpgrade;
    strstr << " (" << id.rawId() << ")";
    return strstr.str();
  }

  if ( subdet == StripSubdetector::TEC ) {
    unsigned int              theWheel  = tecWheel(id);
    unsigned int              theModule = tecModule(id);
    std::vector<unsigned int> thePetal  = tecPetalInfo(id);
    unsigned int              theRing   = tecRing(id);
    std::string side;
    std::string petal;
    side  = (tecSide(id) == 1 ) ? "-" : "+";
    petal = (thePetal[0] == 1 ) ? "back" : "front";
    std::string type;
    type = (isStereo(id)) ? "stereo" : type;
    type = (isRPhi(id)) ? "r-phi" : type;
    type = (isStereo(id) || isRPhi(id)) ? type+" glued": "module";
    std::string typeUpgrade;
    typeUpgrade = (isLower(id)) ? "lower" : typeUpgrade;
    typeUpgrade = (isUpper(id)) ? "upper" : typeUpgrade;
    typeUpgrade = (isUpper(id) || isLower(id)) ? typeUpgrade+" stack": "module";
    strstr << "TEC" << side
	   << " Wheel " << theWheel
	   << " Petal " << thePetal[1] << " " << petal
	   << " Ring " << theRing;
    strstr << " Module for phase0 " << theModule << " " << type;
    strstr << " Module for phase2 " << theModule << " " << typeUpgrade;
    strstr << " (" << id.rawId() << ")";

    return strstr.str();
  }


  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::module";
  return strstr.str();
}

