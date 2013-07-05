#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include <sstream>

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
    type = (tibStereo(id) == 0) ? "r-phi" : "stereo";
    type = (tibGlued(id) == 0) ? type : type+" glued";
    type = (tibIsDoubleSide(id)) ? "double side" : type;
    strstr << "TIB" << side
	   << " Layer " << theLayer << " " << part
	   << " String " << theString[2]
	   << " Module " << theModule << " " << type
	   << " (" << id.rawId() << ")";
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
    type = (tidStereo(id) == 0) ? "r-phi" : "stereo";
    type = (tidGlued(id) == 0) ? type : type+" glued";
    type = (tidIsDoubleSide(id)) ? "double side" : type;
    strstr << "TID" << side
	   << " Disk " << theDisk
	   << " Ring " << theRing << " " << part
	   << " Module " << theModule[1] << " " << type
	   << " (" << id.rawId() << ")";
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
    type = (tobStereo(id) == 0) ? "r-phi" : "stereo";
    type = (tobGlued(id) == 0) ? type : type+" glued";
    type = (tobIsDoubleSide(id)) ? "double side" : type;
    strstr << "TOB" << side
	   << " Layer " << theLayer
	   << " Rod " << theRod[1]
	   << " Module " << theModule << " " << type
	   << " (" << id.rawId() << ")";
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
    type = (tecStereo(id) == 0) ? "r-phi" : "stereo";
    type = (tecGlued(id) == 0) ? type : type+" glued";
    type = (tecIsDoubleSide(id)) ? "double side" : type;
    strstr << "TEC" << side
	   << " Wheel " << theWheel
	   << " Petal " << thePetal[1] << " " << petal
	   << " Ring " << theRing
	   << " Module " << theModule << " " << type
	   << " (" << id.rawId() << ")";

    return strstr.str();
  }


  throw cms::Exception("Invalid DetId") << "Unsupported DetId in TrackerTopology::module";
  return strstr.str();
}

