#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

GeomDetType::GeomDetType( const std::string& n, SubDetector subdet) :
    theName(n), theSubDet(subdet) {}


GeomDetType::~GeomDetType() 
{}


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

bool GeomDetType::isRPC() const
{   
    return (theSubDet == RPCBarrel || theSubDet == RPCEndcap) ;
}

bool GeomDetType::isMuon() const
{
  return ( isRPC() || theSubDet == DT || theSubDet == CSC ) ;
}
