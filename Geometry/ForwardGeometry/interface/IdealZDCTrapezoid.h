#ifndef Geometry_ForwardGeometry_IdealZDCTrapezoid_H
#define Geometry_ForwardGeometry_IdealZDCTrapezoid_H 1

#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

namespace calogeom {
  /** \class IdealZDCTrapezoid
    
  Trapezoid class used for ZDC volumes.  
  
  Required parameters for an ideal zdc trapezoid:
  
  - dz, dx, dy 
  - locaton x, y and z of faceCenter
  - tilt angle of z faces
  
  Total: 7 parameters 
  
  $Date: 2007/08/09 16:38:35 $
  $Revision: 1.1 $
  \author E. Garcia - UIC
  */
  class IdealZDCTrapezoid: public CaloCellGeometry {
  public:
    IdealZDCTrapezoid(const GlobalPoint& faceCenter,float tiltAngle, float deltaX, float deltaY, float deltaZ);
    
    virtual ~IdealZDCTrapezoid() { }
    virtual bool inside(const GlobalPoint & point) const;  
    virtual const std::vector<GlobalPoint> & getCorners() const;
    
  private:
    float deltaX_;
    float deltaY_;
    float deltaZ_;
    float tiltAngle_;
    mutable std::vector<GlobalPoint> points_;
  };
}

#endif
