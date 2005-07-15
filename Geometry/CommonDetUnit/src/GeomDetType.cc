#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
//#include "CommonDet/Topologies/interface/Topology.h"

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
