#include "DataFormats/GeometrySurface/interface/BoundingBox.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"


BoundingBox::BoundingBox(const BoundPlane& plane) {
 float hLen = plane.bounds().length() / 2;
 float hWid = plane.bounds().width() / 2;
 float hThick = plane.bounds().thickness() / 2;

  m_corners[0]  = plane.toGlobal( LocalPoint( hWid, hLen, hThick));
  m_corners[1]  = plane.toGlobal( LocalPoint( hWid, hLen,-hThick));
  m_corners[2]  = plane.toGlobal( LocalPoint( hWid,-hLen, hThick));
  m_corners[3]  = plane.toGlobal( LocalPoint( hWid,-hLen,-hThick));
  m_corners[4]  = plane.toGlobal( LocalPoint(-hWid, hLen, hThick));
  m_corners[5]  = plane.toGlobal( LocalPoint(-hWid, hLen,-hThick));
  m_corners[6]  = plane.toGlobal( LocalPoint(-hWid,-hLen, hThick));
  m_corners[7]  = plane.toGlobal( LocalPoint(-hWid,-hLen,-hThick));
 
}


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
