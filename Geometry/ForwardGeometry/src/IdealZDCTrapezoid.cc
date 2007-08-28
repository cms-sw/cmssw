#include "Geometry/ForwardGeometry/interface/IdealZDCTrapezoid.h"
#include <math.h>

namespace calogeom {
  
  IdealZDCTrapezoid::IdealZDCTrapezoid(const GlobalPoint& faceCenter, float tiltAngle, float dx, float dy, float dz):
    CaloCellGeometry(faceCenter),
    deltaX_(dx),
    deltaY_(dy),
    deltaZ_(dz),
    tiltAngle_(tiltAngle){ }
  
  const std::vector<GlobalPoint> & IdealZDCTrapezoid::getCorners() const {
    if (points_.empty()) {
      float z1 = 0;
      float z2 = 0;
      GlobalPoint p=getPosition();
      if(p.z() >= 0){
	z1 = p.z()- cos(tiltAngle_)*deltaY_/2.;
	z2 = p.z()+ cos(tiltAngle_)*deltaY_/2.;
      } 
      else{
	z1 = p.z()+ cos(tiltAngle_)*deltaY_/2.;
	z2 = p.z()- cos(tiltAngle_)*deltaY_/2.;
      } 
      float z3 = z1 + deltaZ_;
      float z4 = z2 + deltaZ_;
      float x1 =  deltaX_/2;
      float x2 = -deltaX_/2;
      float y1 = p.y() + sin(tiltAngle_)*deltaY_/2;
      float y2 = p.y() - sin(tiltAngle_)*deltaY_/2;
      points_.push_back(GlobalPoint(x1,y1,z1)); 
      points_.push_back(GlobalPoint(x2,y1,z1));
      points_.push_back(GlobalPoint(x2,y2,z2));
      points_.push_back(GlobalPoint(x1,y2,z2));
      points_.push_back(GlobalPoint(x1,y1,z3));
      points_.push_back(GlobalPoint(x2,y1,z3));
      points_.push_back(GlobalPoint(x2,y2,z4));
      points_.push_back(GlobalPoint(x1,y2,z4));
    }
    return points_;
  }

  bool IdealZDCTrapezoid::inside(const GlobalPoint& point) const {
    bool is_inside=true;
    float m = 0;
    float blow =0;
    float bhigh =0;
    const GlobalPoint& face = getPosition();
    // x
    is_inside=is_inside && (fabs(point.x() - face.x()) <= deltaX_/2);
    // y
    is_inside=is_inside && (fabs(point.y() - face.y()) <= (sin(tiltAngle_)*deltaY_/2));
    // z 
    m = tan(tiltAngle_);
    blow = face.y() - m*face.z();
    if(point.z()>0){
      bhigh = face.y() - m*(face.z() + deltaZ_);
      is_inside=is_inside && (point.z()> (point.y()- blow)/m)  && (point.z()<=((point.y()-bhigh)/m));
    } 
    else {
      m = -m;
      bhigh = face.y() - m*(face.z() - deltaZ_);
      is_inside=is_inside && (point.z()> (point.y()- blow)/m)  && (point.z()<=((point.y()-bhigh)/m));
    }
    return is_inside;
  }
}
