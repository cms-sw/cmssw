#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>

TrackerTopology::TrackerTopology(const PixelBarrelValues &pxb,
                                 const PixelEndcapValues &pxf,
                                 const TECValues &tecv,
                                 const TIBValues &tibv,
                                 const TIDValues &tidv,
                                 const TOBValues &tobv)
    : pbVals_(pxb),
      pfVals_(pxf),
      tobVals_(tobv),
      tibVals_(tibv),
      tidVals_(tidv),
      tecVals_(tecv),
      bits_per_field{[PBModule] = {pbVals_.moduleStartBit_, pbVals_.moduleMask_, PixelSubdetector::PixelBarrel},
                     [PBLadder] = {pbVals_.ladderStartBit_, pbVals_.ladderMask_, PixelSubdetector::PixelBarrel},
                     [PBLayer] = {pbVals_.layerStartBit_, pbVals_.layerMask_, PixelSubdetector::PixelBarrel},
                     [PFModule] = {pfVals_.moduleStartBit_, pfVals_.moduleMask_, PixelSubdetector::PixelEndcap},
                     [PFPanel] = {pfVals_.panelStartBit_, pfVals_.panelMask_, PixelSubdetector::PixelEndcap},
                     [PFBlade] = {pfVals_.bladeStartBit_, pfVals_.bladeMask_, PixelSubdetector::PixelEndcap},
                     [PFDisk] = {pfVals_.diskStartBit_, pfVals_.diskMask_, PixelSubdetector::PixelEndcap},
                     [PFSide] = {pfVals_.sideStartBit_, pfVals_.sideMask_, PixelSubdetector::PixelEndcap}} {}

unsigned int TrackerTopology::side(const DetId &id) const {
  uint32_t subdet = id.subdetId();
  if (subdet == PixelSubdetector::PixelBarrel)
    return 0;
  if (subdet == PixelSubdetector::PixelEndcap)
    return pxfSide(id);
  if (subdet == SiStripSubdetector::TIB)
    return 0;
  if (subdet == SiStripSubdetector::TID)
    return tidSide(id);
  if (subdet == SiStripSubdetector::TOB)
    return 0;
  if (subdet == SiStripSubdetector::TEC)
    return tecSide(id);

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::side";
  return 0;
}

unsigned int TrackerTopology::layer(const DetId &id) const {
  uint32_t subdet = id.subdetId();
  if (subdet == PixelSubdetector::PixelBarrel)
    return pxbLayer(id);
  if (subdet == PixelSubdetector::PixelEndcap)
    return pxfDisk(id);
  if (subdet == SiStripSubdetector::TIB)
    return tibLayer(id);
  if (subdet == SiStripSubdetector::TID)
    return tidWheel(id);
  if (subdet == SiStripSubdetector::TOB)
    return tobLayer(id);
  if (subdet == SiStripSubdetector::TEC)
    return tecWheel(id);

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::layer";
  return 0;
}

unsigned int TrackerTopology::module(const DetId &id) const {
  uint32_t subdet = id.subdetId();
  if (subdet == PixelSubdetector::PixelBarrel)
    return pxbModule(id);
  if (subdet == PixelSubdetector::PixelEndcap)
    return pxfModule(id);
  if (subdet == SiStripSubdetector::TIB)
    return tibModule(id);
  if (subdet == SiStripSubdetector::TID)
    return tidModule(id);
  if (subdet == SiStripSubdetector::TOB)
    return tobModule(id);
  if (subdet == SiStripSubdetector::TEC)
    return tecModule(id);

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::module";
  return 0;
}

uint32_t TrackerTopology::glued(const DetId &id) const {
  uint32_t subdet = id.subdetId();
  if (subdet == PixelSubdetector::PixelBarrel)
    return 0;
  if (subdet == PixelSubdetector::PixelEndcap)
    return 0;
  if (subdet == SiStripSubdetector::TIB)
    return tibGlued(id);
  if (subdet == SiStripSubdetector::TID)
    return tidGlued(id);
  if (subdet == SiStripSubdetector::TOB)
    return tobGlued(id);
  if (subdet == SiStripSubdetector::TEC)
    return tecGlued(id);

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::glued";
  return 0;
}

uint32_t TrackerTopology::stack(const DetId &id) const {
  uint32_t subdet = id.subdetId();
  if (subdet == PixelSubdetector::PixelBarrel)
    return 0;
  if (subdet == PixelSubdetector::PixelEndcap)
    return 0;
  if (subdet == SiStripSubdetector::TIB)
    return tibStack(id);
  if (subdet == SiStripSubdetector::TID)
    return tidStack(id);
  if (subdet == SiStripSubdetector::TOB)
    return tobStack(id);
  if (subdet == SiStripSubdetector::TEC)
    return tecStack(id);

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::stack";
}

uint32_t TrackerTopology::doubleSensor(const DetId &id) const {
  uint32_t subdet = id.subdetId();
  if (subdet == PixelSubdetector::PixelBarrel)
    return pixDouble(id);
  if (subdet == PixelSubdetector::PixelEndcap)
    return 0;
  if (subdet == SiStripSubdetector::TIB)
    return 0;
  if (subdet == SiStripSubdetector::TID)
    return 0;
  if (subdet == SiStripSubdetector::TOB)
    return 0;
  if (subdet == SiStripSubdetector::TEC)
    return 0;

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::stack";
}

uint32_t TrackerTopology::first(const DetId &id) const {
  uint32_t subdet = id.subdetId();
  if (subdet == PixelSubdetector::PixelBarrel)
    return pixFirst(id);
  if (subdet == PixelSubdetector::PixelEndcap)
    return 0;
  if (subdet == SiStripSubdetector::TIB)
    return 0;
  if (subdet == SiStripSubdetector::TID)
    return 0;
  if (subdet == SiStripSubdetector::TOB)
    return 0;
  if (subdet == SiStripSubdetector::TEC)
    return 0;

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::doubleSens";
}

uint32_t TrackerTopology::second(const DetId &id) const {
  uint32_t subdet = id.subdetId();
  if (subdet == PixelSubdetector::PixelBarrel)
    return pixSecond(id);
  if (subdet == PixelSubdetector::PixelEndcap)
    return 0;
  if (subdet == SiStripSubdetector::TIB)
    return 0;
  if (subdet == SiStripSubdetector::TID)
    return 0;
  if (subdet == SiStripSubdetector::TOB)
    return 0;
  if (subdet == SiStripSubdetector::TEC)
    return 0;

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::doubleSensor";
}

uint32_t TrackerTopology::lower(const DetId &id) const {
  uint32_t subdet = id.subdetId();
  if (subdet == PixelSubdetector::PixelBarrel)
    return 0;
  if (subdet == PixelSubdetector::PixelEndcap)
    return 0;
  if (subdet == SiStripSubdetector::TIB)
    return tibLower(id);
  if (subdet == SiStripSubdetector::TID)
    return tidLower(id);
  if (subdet == SiStripSubdetector::TOB)
    return tobLower(id);
  if (subdet == SiStripSubdetector::TEC)
    return tecLower(id);

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::lower";
}

uint32_t TrackerTopology::upper(const DetId &id) const {
  uint32_t subdet = id.subdetId();
  if (subdet == PixelSubdetector::PixelBarrel)
    return 0;
  if (subdet == PixelSubdetector::PixelEndcap)
    return 0;
  if (subdet == SiStripSubdetector::TIB)
    return tibUpper(id);
  if (subdet == SiStripSubdetector::TID)
    return tidUpper(id);
  if (subdet == SiStripSubdetector::TOB)
    return tobUpper(id);
  if (subdet == SiStripSubdetector::TEC)
    return tecUpper(id);

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::upper";
}

bool TrackerTopology::isStereo(const DetId &id) const {
  uint32_t subdet = id.subdetId();
  if (subdet == PixelSubdetector::PixelBarrel)
    return false;
  if (subdet == PixelSubdetector::PixelEndcap)
    return false;
  if (subdet == SiStripSubdetector::TIB)
    return tibStereo(id) != 0;
  if (subdet == SiStripSubdetector::TID)
    return tidStereo(id) != 0;
  if (subdet == SiStripSubdetector::TOB)
    return tobStereo(id) != 0;
  if (subdet == SiStripSubdetector::TEC)
    return tecStereo(id) != 0;

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::isStereo";
  return false;
}

bool TrackerTopology::isRPhi(const DetId &id) const {
  uint32_t subdet = id.subdetId();
  if (subdet == PixelSubdetector::PixelBarrel)
    return false;
  if (subdet == PixelSubdetector::PixelEndcap)
    return false;
  if (subdet == SiStripSubdetector::TIB)
    return tibRPhi(id) != 0;
  if (subdet == SiStripSubdetector::TID)
    return tidRPhi(id) != 0;
  if (subdet == SiStripSubdetector::TOB)
    return tobRPhi(id) != 0;
  if (subdet == SiStripSubdetector::TEC)
    return tecRPhi(id) != 0;

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::isRPhi";
  return false;
}

bool TrackerTopology::isDoubleSens(const DetId &id) const {
  uint32_t subdet = id.subdetId();
  if (subdet == PixelSubdetector::PixelBarrel)
    return pixDouble(id) != 0;
  if (subdet == PixelSubdetector::PixelEndcap)
    return false;
  if (subdet == SiStripSubdetector::TIB)
    return false;
  if (subdet == SiStripSubdetector::TID)
    return false;
  if (subdet == SiStripSubdetector::TOB)
    return false;
  if (subdet == SiStripSubdetector::TEC)
    return false;

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::isDoubleSens";
  return false;
}

bool TrackerTopology::isLower(const DetId &id) const {
  uint32_t subdet = id.subdetId();
  if (subdet == PixelSubdetector::PixelBarrel)
    return false;
  if (subdet == PixelSubdetector::PixelEndcap)
    return false;
  if (subdet == SiStripSubdetector::TIB)
    return tibLower(id) != 0;
  if (subdet == SiStripSubdetector::TID)
    return tidLower(id) != 0;
  if (subdet == SiStripSubdetector::TOB)
    return tobLower(id) != 0;
  if (subdet == SiStripSubdetector::TEC)
    return tecLower(id) != 0;

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::isLower";
  return false;
}

bool TrackerTopology::isUpper(const DetId &id) const {
  uint32_t subdet = id.subdetId();
  if (subdet == PixelSubdetector::PixelBarrel)
    return false;
  if (subdet == PixelSubdetector::PixelEndcap)
    return false;
  if (subdet == SiStripSubdetector::TIB)
    return tibUpper(id) != 0;
  if (subdet == SiStripSubdetector::TID)
    return tidUpper(id) != 0;
  if (subdet == SiStripSubdetector::TOB)
    return tobUpper(id) != 0;
  if (subdet == SiStripSubdetector::TEC)
    return tecUpper(id) != 0;

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::isUpper";
  return false;
}

bool TrackerTopology::isFirst(const DetId &id) const {
  uint32_t subdet = id.subdetId();
  if (subdet == PixelSubdetector::PixelBarrel)
    return pixFirst(id) != 0;
  if (subdet == PixelSubdetector::PixelEndcap)
    return false;
  if (subdet == SiStripSubdetector::TIB)
    return false;
  if (subdet == SiStripSubdetector::TID)
    return false;
  if (subdet == SiStripSubdetector::TOB)
    return false;
  if (subdet == SiStripSubdetector::TEC)
    return false;

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::isFirst";
  return false;
}

bool TrackerTopology::isSecond(const DetId &id) const {
  uint32_t subdet = id.subdetId();
  if (subdet == PixelSubdetector::PixelBarrel)
    return pixSecond(id) != 0;
  if (subdet == PixelSubdetector::PixelEndcap)
    return false;
  if (subdet == SiStripSubdetector::TIB)
    return false;
  if (subdet == SiStripSubdetector::TID)
    return false;
  if (subdet == SiStripSubdetector::TOB)
    return false;
  if (subdet == SiStripSubdetector::TEC)
    return false;

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::isSecond";
  return false;
}

DetId TrackerTopology::partnerDetId(const DetId &id) const {
  uint32_t subdet = id.subdetId();
  if (subdet == PixelSubdetector::PixelBarrel)
    return pixPartnerDetId(id);
  if (subdet == PixelSubdetector::PixelEndcap)
    return 0;
  if (subdet == SiStripSubdetector::TIB)
    return tibPartnerDetId(id);
  if (subdet == SiStripSubdetector::TID)
    return tidPartnerDetId(id);
  if (subdet == SiStripSubdetector::TOB)
    return tobPartnerDetId(id);
  if (subdet == SiStripSubdetector::TEC)
    return tecPartnerDetId(id);

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::partnerDetId";
  return 0;
}

std::string TrackerTopology::print(DetId id) const {
  uint32_t subdet = id.subdetId();
  std::stringstream strstr;

  if (subdet == PixelSubdetector::PixelBarrel) {
    unsigned int theLayer = pxbLayer(id);
    unsigned int theLadder = pxbLadder(id);
    unsigned int theModule = pxbModule(id);
    std::string typeUpgrade;
    typeUpgrade = (isFirst(id)) ? "first" : typeUpgrade;
    typeUpgrade = (isSecond(id)) ? "second" : typeUpgrade;
    typeUpgrade = (isFirst(id) || isSecond(id)) ? typeUpgrade + " double" : "module";
    strstr << "PixelBarrel"
           << " Layer " << theLayer << " Ladder " << theLadder;
    strstr << " Module for phase0 " << theModule;
    strstr << " Module for phase2 " << theModule << " " << typeUpgrade;
    strstr << " (" << id.rawId() << ")";
    return strstr.str();
  }

  if (subdet == PixelSubdetector::PixelEndcap) {
    unsigned int theSide = pxfSide(id);
    unsigned int theDisk = pxfDisk(id);
    unsigned int theBlade = pxfBlade(id);
    unsigned int thePanel = pxfPanel(id);
    unsigned int theModule = pxfModule(id);
    std::string side = (pxfSide(id) == 1) ? "-" : "+";
    strstr << "PixelEndcap"
           << " Side   " << theSide << side << " Disk   " << theDisk << " Blade  " << theBlade << " Panel  " << thePanel
           << " Module " << theModule;
    strstr << " (" << id.rawId() << ")";
    return strstr.str();
  }

  if (subdet == SiStripSubdetector::TIB) {
    unsigned int theLayer = tibLayer(id);
    std::vector<unsigned int> theString = tibStringInfo(id);
    unsigned int theModule = tibModule(id);
    std::string side;
    std::string part;
    side = (theString[0] == 1) ? "-" : "+";
    part = (theString[1] == 1) ? "int" : "ext";
    std::string type;
    type = (isStereo(id)) ? "stereo" : type;
    type = (isRPhi(id)) ? "r-phi" : type;
    type = (isStereo(id) || isRPhi(id)) ? type + " glued" : "module";
    std::string typeUpgrade;
    typeUpgrade = (isLower(id)) ? "lower" : typeUpgrade;
    typeUpgrade = (isUpper(id)) ? "upper" : typeUpgrade;
    typeUpgrade = (isUpper(id) || isLower(id)) ? typeUpgrade + " stack" : "module";
    strstr << "TIB" << side << " Layer " << theLayer << " " << part << " String " << theString[2];
    strstr << " Module for phase0 " << theModule << " " << type;
    strstr << " Module for phase2 " << theModule << " " << typeUpgrade;
    strstr << " (" << id.rawId() << ")";
    return strstr.str();
  }

  if (subdet == SiStripSubdetector::TID) {
    unsigned int theSide = tidSide(id);
    unsigned int theWheel = tidWheel(id);
    unsigned int theRing = tidRing(id);
    std::vector<unsigned int> theModule = tidModuleInfo(id);
    std::string side;
    std::string part;
    side = (tidSide(id) == 1) ? "-" : "+";
    part = (theModule[0] == 1) ? "back" : "front";
    std::string type;
    type = (isStereo(id)) ? "stereo" : type;
    type = (isRPhi(id)) ? "r-phi" : type;
    type = (isStereo(id) || isRPhi(id)) ? type + " glued" : "module";
    std::string typeUpgrade;
    typeUpgrade = (isLower(id)) ? "lower" : typeUpgrade;
    typeUpgrade = (isUpper(id)) ? "upper" : typeUpgrade;
    typeUpgrade = (isUpper(id) || isLower(id)) ? typeUpgrade + " stack" : "module";
    strstr << "TID"
           << " Side   " << theSide << side << " Wheel " << theWheel << " Ring " << theRing << " " << part;
    strstr << " Module for phase0 " << theModule[1] << " " << type;
    strstr << " Module for phase2 " << theModule[1] << " " << typeUpgrade;
    strstr << " (" << id.rawId() << ")";
    return strstr.str();
  }

  if (subdet == SiStripSubdetector::TOB) {
    unsigned int theLayer = tobLayer(id);
    std::vector<unsigned int> theRod = tobRodInfo(id);
    unsigned int theModule = tobModule(id);
    std::string side;
    std::string part;
    side = (((theRod[0] == 1) ? "-" : ((theRod[0] == 2) ? "+" : (theRod[0] == 3) ? "0" : "")));
    //    side = (theRod[0] == 2 ) ? "+" : "";
    //    side = (theRod[0] == 3 ) ? "0" : "";
    std::string type;
    type = (isStereo(id)) ? "stereo" : type;
    type = (isRPhi(id)) ? "r-phi" : type;
    type = (isStereo(id) || isRPhi(id)) ? type + " glued" : "module";
    std::string typeUpgrade;
    typeUpgrade = (isLower(id)) ? "lower" : typeUpgrade;
    typeUpgrade = (isUpper(id)) ? "upper" : typeUpgrade;
    typeUpgrade = (isUpper(id) || isLower(id)) ? typeUpgrade + " stack" : "module";
    strstr << "TOB" << side << " Layer " << theLayer << " Rod " << theRod[1];
    strstr << " Module for phase0 " << theModule << " " << type;
    strstr << " Module for phase2 " << theModule << " " << typeUpgrade;
    strstr << " (" << id.rawId() << ")";
    return strstr.str();
  }

  if (subdet == SiStripSubdetector::TEC) {
    unsigned int theSide = tecSide(id);
    unsigned int theWheel = tecWheel(id);
    unsigned int theModule = tecModule(id);
    std::vector<unsigned int> thePetal = tecPetalInfo(id);
    unsigned int theRing = tecRing(id);
    std::string side;
    std::string petal;
    side = (tecSide(id) == 1) ? "-" : "+";
    petal = (thePetal[0] == 1) ? "back" : "front";
    std::string type;
    type = (isStereo(id)) ? "stereo" : type;
    type = (isRPhi(id)) ? "r-phi" : type;
    type = (isStereo(id) || isRPhi(id)) ? type + " glued" : "module";
    std::string typeUpgrade;
    typeUpgrade = (isLower(id)) ? "lower" : typeUpgrade;
    typeUpgrade = (isUpper(id)) ? "upper" : typeUpgrade;
    typeUpgrade = (isUpper(id) || isLower(id)) ? typeUpgrade + " stack" : "module";
    strstr << "TEC"
           << " Side   " << theSide << side << " Wheel " << theWheel << " Petal " << thePetal[1] << " " << petal
           << " Ring " << theRing;
    strstr << " Module for phase0 " << theModule << " " << type;
    strstr << " Module for phase2 " << theModule << " " << typeUpgrade;
    strstr << " (" << id.rawId() << ")";

    return strstr.str();
  }

  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::module";
  return strstr.str();
}

SiStripModuleGeometry TrackerTopology::moduleGeometry(const DetId &id) const {
  switch (id.subdetId()) {
    case SiStripSubdetector::TIB:
      return tibLayer(id) < 3 ? SiStripModuleGeometry::IB1 : SiStripModuleGeometry::IB2;
    case SiStripSubdetector::TOB:
      return tobLayer(id) < 5 ? SiStripModuleGeometry::OB2 : SiStripModuleGeometry::OB1;
    case SiStripSubdetector::TID:
      switch (tidRing(id)) {
        case 1:
          return SiStripModuleGeometry::W1A;
        case 2:
          return SiStripModuleGeometry::W2A;
        case 3:
          return SiStripModuleGeometry::W3A;
      }
      return SiStripModuleGeometry::UNKNOWNGEOMETRY;
    case SiStripSubdetector::TEC:
      switch (tecRing(id)) {
        case 1:
          return SiStripModuleGeometry::W1B;
        case 2:
          return SiStripModuleGeometry::W2B;
        case 3:
          return SiStripModuleGeometry::W3B;
        case 4:
          return SiStripModuleGeometry::W4;
          //generic function to return DetIds and boolean factors
        case 5:
          return SiStripModuleGeometry::W5;
        case 6:
          return SiStripModuleGeometry::W6;
        case 7:
          return SiStripModuleGeometry::W7;
      }
  }
  return SiStripModuleGeometry::UNKNOWNGEOMETRY;
}
int TrackerTopology::getOTLayerNumber(const DetId &id) const {
  int layer = -1;

  if (id.det() == DetId::Tracker) {
    if (id.subdetId() == SiStripSubdetector::TOB) {
      layer = tobLayer(id);
    } else if (id.subdetId() == SiStripSubdetector::TID) {
      layer = 100 * tidSide(id) + tidWheel(id);
    } else {
      edm::LogInfo("TrackerTopology") << ">>> Invalid subdetId()  ";
    }
  }
  return layer;
}

int TrackerTopology::getITPixelLayerNumber(const DetId &id) const {
  int layer = -1;

  if (id.det() == DetId::Tracker) {
    if (id.subdetId() == PixelSubdetector::PixelBarrel) {
      layer = pxbLayer(id);
    } else if (id.subdetId() == PixelSubdetector::PixelEndcap) {
      layer = 100 * pxfSide(id) + pxfDisk(id);
    } else {
      edm::LogInfo("TrackerTopology") << ">>> Invalid subdetId()  ";
    }
  }
  return layer;
}
