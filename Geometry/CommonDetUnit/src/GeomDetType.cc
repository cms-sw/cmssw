#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

using namespace GeomDetEnumerators;

GeomDetType::GeomDetType( const std::string& n, SubDetector subdet) :
  theName(n), theSubDet(subdet) {}


GeomDetType::~GeomDetType() 
{}


bool GeomDetType::isBarrel() const
{
  return (theSubDet == PixelBarrel || theSubDet == TIB || theSubDet == TOB || isDT() || theSubDet == RPCBarrel);
}

bool GeomDetType::isEndcap() const
{
  return (!isBarrel());
}


bool GeomDetType::isTrackerStrip() const
{
  return (theSubDet == TIB || theSubDet == TOB || 
	  theSubDet == TID || theSubDet == TEC);
}

bool GeomDetType::isTrackerPixel() const
{
  return (theSubDet == PixelBarrel || theSubDet == PixelEndcap); 
}

bool GeomDetType::isTracker() const
{
  return ( isTrackerStrip() || isTrackerPixel() );
}


bool GeomDetType::isDT() const
{   
  return (theSubDet == DT) ;
}

bool GeomDetType::isCSC() const
{   
  return (theSubDet == CSC) ;
}


bool GeomDetType::isRPC() const
{   
  return (theSubDet == RPCBarrel || theSubDet == RPCEndcap) ;
}

bool GeomDetType::isGEM() const
{   
  return (theSubDet == GEM ) ;
}

bool GeomDetType::isME0() const
{   
  return (theSubDet == ME0 ) ;
}


bool GeomDetType::isMuon() const
{
  return (theSubDet == DT || theSubDet == CSC || isRPC() || theSubDet == GEM || theSubDet == ME0) ;
}
