#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include<ostream>
using namespace GeomDetEnumerators;

std::ostream& operator<<( std::ostream& s, Location l) {
  if (l == barrel) return s << "barrel";
  else return s << "endcap";
}

std::ostream& operator<<( std::ostream& s, SubDetector m){
  if ( m == PixelBarrel ) return s << "PixelBarrel";
  else if ( m == PixelEndcap ) return s << "PixelEndcap";
  else if ( m == TIB) return s << "TIB";
  else if (m == TOB) return s << "TOB";
  else if (m == TID) return s << "TID";
  else if (m == TEC) return s << "TEC";
  else if ( m == DT ) return s << "DT";
  else if ( m == CSC ) return s << "CSC";
  else if ( m == RPCBarrel ) return s << "RPCBarrel";
  else if ( m == RPCEndcap ) return s << "RPCEndcap";
  else if ( m == GEM) return s << "GEM";
  else if ( m == ME0 ) return s << "ME0";
  else if ( m == P2OTB ) return s << "Phase2OTBarrel";
  else if ( m == P2OTEC ) return s << "Phase2OTEndcap";
  else if ( m == P1PXB ) return s << "Phase1PixelBarrel";
  else if ( m == P1PXEC ) return s << "Phase1PixelEndcap";
  else if ( m == P2PXEC ) return s << "Phase2PixelEndcap";
  else return s << "?";
}


bool GeomDetEnumerators::isBarrel(const GeomDetEnumerators::SubDetector subdet)
{
  return (subdet == PixelBarrel || subdet == TIB || subdet == TOB || subdet == P1PXB || subdet == P2OTB || isDT(subdet) || subdet == RPCBarrel);
}

bool GeomDetEnumerators::isEndcap(const GeomDetEnumerators::SubDetector subdet)
{
  return (!isBarrel(subdet));
}


bool GeomDetEnumerators::isTrackerStrip(const GeomDetEnumerators::SubDetector subdet)
{
  return (subdet == TIB || subdet == TOB ||
	  subdet == TID || subdet == TEC);
}

bool GeomDetEnumerators::isTrackerPixel(const GeomDetEnumerators::SubDetector subdet)
{
  return (subdet == PixelBarrel || subdet == PixelEndcap || 
	  subdet == P1PXB || subdet == P1PXEC || subdet == P2PXEC ||
	  subdet == P2OTB || subdet == P2OTEC); 
}

bool GeomDetEnumerators::isTracker(const GeomDetEnumerators::SubDetector subdet)
{
  return ( isTrackerStrip(subdet) || isTrackerPixel(subdet) );
}


bool GeomDetEnumerators::isDT(const GeomDetEnumerators::SubDetector subdet)
{   
  return (subdet == DT) ;
}

bool GeomDetEnumerators::isCSC(const GeomDetEnumerators::SubDetector subdet)
{   
  return (subdet == CSC) ;
}


bool GeomDetEnumerators::isRPC(const GeomDetEnumerators::SubDetector subdet)
{   
  return (subdet == RPCBarrel || subdet == RPCEndcap) ;
}

bool GeomDetEnumerators::isGEM(const GeomDetEnumerators::SubDetector subdet)
{   
  return (subdet == GEM ) ;
}

bool GeomDetEnumerators::isME0(const GeomDetEnumerators::SubDetector subdet)
{   
  return (subdet == ME0 ) ;
}


bool GeomDetEnumerators::isMuon(const GeomDetEnumerators::SubDetector subdet)
{
  return (subdet == DT || subdet == CSC || isRPC(subdet) || subdet == GEM || subdet == ME0) ;
}
