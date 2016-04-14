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

uint32_t TrackerTopology::glued(const DetId &id) const {

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

    throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::glued";
    return 0;
}

uint32_t TrackerTopology::stack(const DetId &id) const {

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

    throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::stack";
}

uint32_t TrackerTopology::lower(const DetId &id) const {

    uint32_t subdet=id.subdetId();
    if ( subdet == PixelSubdetector::PixelBarrel )
      return 0;
    if ( subdet == PixelSubdetector::PixelEndcap )
      return 0;
    if ( subdet == StripSubdetector::TIB )
      return tibLower(id);
    if ( subdet == StripSubdetector::TID )
      return tidLower(id);
    if ( subdet == StripSubdetector::TOB )
      return tobLower(id);
    if ( subdet == StripSubdetector::TEC )
      return tecLower(id);

    throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::lower";
}

uint32_t TrackerTopology::upper(const DetId &id) const {

    uint32_t subdet=id.subdetId();
    if ( subdet == PixelSubdetector::PixelBarrel )
      return 0;
    if ( subdet == PixelSubdetector::PixelEndcap )
      return 0;
    if ( subdet == StripSubdetector::TIB )
      return tibUpper(id);
    if ( subdet == StripSubdetector::TID )
      return tidUpper(id);
    if ( subdet == StripSubdetector::TOB )
      return tobUpper(id);
    if ( subdet == StripSubdetector::TEC )
      return tecUpper(id);

    throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::upper";
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

DetId TrackerTopology::partnerDetId(const DetId &id) const {

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

    throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::partnerDetId";
    return 0;
}

std::string TrackerTopology::print(DetId id) const {
  uint32_t subdet=id.subdetId();
  std::stringstream strstr;

  if ( subdet == PixelSubdetector::PixelBarrel ) {
    unsigned int theLayer  = pxbLayer(id);
    unsigned int theLadder = pxbLadder(id);
    unsigned int theModule = pxbModule(id);
    strstr << "PixelBarrel" 
	   << " Layer " << theLayer
	   << " Ladder " << theLadder
           << " Module " << theModule ;
    strstr << " (" << id.rawId() << ")";
    return strstr.str();
  }

  if ( subdet == PixelSubdetector::PixelEndcap ) {
    unsigned int theSide   = pxfSide(id);
    unsigned int theDisk   = pxfDisk(id);
    unsigned int theBlade  = pxfBlade(id);
    unsigned int thePanel  = pxfPanel(id);
    unsigned int theModule = pxfModule(id);
    std::string side  = (pxfSide(id) == 1 ) ? "-" : "+";
    strstr << "PixelEndcap" 
           << " Side   " << theSide << side
	   << " Disk   " << theDisk
	   << " Blade  " << theBlade
	   << " Panel  " << thePanel
           << " Module " << theModule ;
    strstr << " (" << id.rawId() << ")";
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
    unsigned int 	 theSide   = tidSide(id);
    unsigned int         theWheel  = tidWheel(id);
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
    strstr << "TID" 
           << " Side   " << theSide << side
	   << " Wheel " << theWheel
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
    side = (((theRod[0] == 1 ) ? "-" : ((theRod[0] == 2 ) ? "+" : (theRod[0] == 3 ) ? "0" : "")));
//    side = (theRod[0] == 2 ) ? "+" : "";
//    side = (theRod[0] == 3 ) ? "0" : "";
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
    unsigned int 	      theSide   = tecSide(id);
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
    strstr << "TEC" 
           << " Side   " << theSide << side
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


SiStripDetId::ModuleGeometry TrackerTopology::moduleGeometry(const DetId &id) const {
  switch(id.subdetId()) {
  case StripSubdetector::TIB: return tibLayer(id)<3? SiStripDetId::IB1 : SiStripDetId::IB2;
  case StripSubdetector::TOB: return tobLayer(id)<5? SiStripDetId::OB2 : SiStripDetId::OB1;
  case StripSubdetector::TID: switch (tidRing(id)) {
    case 1: return SiStripDetId::W1A;
    case 2: return SiStripDetId::W2A;
    case 3: return SiStripDetId::W3A;
    }
  case StripSubdetector::TEC: switch (tecRing(id)) {
    case 1: return SiStripDetId::W1B;
    case 2: return SiStripDetId::W2B;
    case 3: return SiStripDetId::W3B;
    case 4: return SiStripDetId::W4;
  //generic function to return DetIds and boolean factors
    case 5: return SiStripDetId::W5;
    case 6: return SiStripDetId::W6;
    case 7: return SiStripDetId::W7;
    }
  }
  return SiStripDetId::UNKNOWNGEOMETRY;
}
