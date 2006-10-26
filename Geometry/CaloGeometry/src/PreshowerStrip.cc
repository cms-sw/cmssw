#include "Geometry/CaloGeometry/interface/PreshowerStrip.h"
//#include <algorithm>
#include <iostream>
//#include "assert.h"

using namespace std;

//----------------------------------------------------------------------

PreshowerStrip::PreshowerStrip()
 {}

//----------------------------------------------------------------------

PreshowerStrip::PreshowerStrip(double dx, double dy, double dz)
{

  dx_ = dx;
  dy_ = dy;
  dz_ = dz;

  corners.resize(8);
  
  corners[0] = GlobalPoint(-dx , -dy , -dz); // (-,-,-)
  corners[1] = GlobalPoint(-dx ,  dy , -dz); // (-,+,-)
  corners[2] = GlobalPoint( dx ,  dy , -dz); // (+,+,-)
  corners[3] = GlobalPoint( dx , -dy , -dz); // (+,-,-)
                           
  corners[4] = GlobalPoint(-dx , -dy , dz);  // (-,-,+)
  corners[5] = GlobalPoint(-dx ,  dy , dz);  // (-,+,+)
  corners[6] = GlobalPoint( dx ,  dy , dz);  // (+,+,+)
  corners[7] = GlobalPoint( dx , -dy , dz);  // (+,-,+)


  // set the reference position as the geometric center of the box
  HepGeom::Point3D<double> position;
  position == HepGeom::Point3D<double>(0.,0.,0.);
  setPosition(GlobalPoint(position.x(),position.y(),position.z()));
}

bool PreshowerStrip::inside(const GlobalPoint & Point) const
{

  const GlobalPoint& center = getPosition();
  if ( abs(Point.x()-center.x()) > dx_ || abs(Point.y()-center.y()) > dy_ || abs(Point.z()-center.z()) > dz_ ) return false;
  return true;
}


const vector<GlobalPoint> & PreshowerStrip::getCorners() const
{ return corners ; }

void PreshowerStrip::hepTransform(const HepTransform3D &transformation)
{

  unsigned int i;

  //Updating corners
  for (i=0; i<corners.size(); ++i)
    {
      HepGeom::Point3D<float> newCorner(corners[i].x(),corners[i].y(),corners[i].z());
      newCorner.transform(transformation);
      corners[i]=GlobalPoint(newCorner.x(),newCorner.y(),newCorner.z());
    }

  //Updating reference position
  const GlobalPoint& position_=getPosition();
  HepGeom::Point3D<float> newPosition(position_.x(),position_.y(),position_.z());
  newPosition.transform(transformation);
  setPosition(GlobalPoint(newPosition.x(),newPosition.y(),newPosition.z()));

}
