
#include "DataFormats/GeometrySurface/interface/BoundingBox.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

std::vector<GlobalPoint>
BoundingBox::corners( const BoundPlane& plane) 
{
   std::vector<GlobalPoint> result;
   result.reserve(8);

  float hLen = plane.bounds().length() / 2;
  float hWid = plane.bounds().width() / 2;
  float hThick = plane.bounds().thickness() / 2;
  
  result.push_back( plane.toGlobal( LocalPoint( hWid, hLen, hThick)));
  result.push_back( plane.toGlobal( LocalPoint( hWid, hLen,-hThick)));
  result.push_back( plane.toGlobal( LocalPoint( hWid,-hLen, hThick)));
  result.push_back( plane.toGlobal( LocalPoint( hWid,-hLen,-hThick)));
  result.push_back( plane.toGlobal( LocalPoint(-hWid, hLen, hThick)));
  result.push_back( plane.toGlobal( LocalPoint(-hWid, hLen,-hThick)));
  result.push_back( plane.toGlobal( LocalPoint(-hWid,-hLen, hThick)));
  result.push_back( plane.toGlobal( LocalPoint(-hWid,-hLen,-hThick)));

  return result;
}
