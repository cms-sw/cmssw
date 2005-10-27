#ifndef GEOMETRY_CALOGEOMETRY_CALOCELLGEOMETRY_H
#define GEOMETRY_CALOGEOMETRY_CALOCELLGEOMETRY_H 1

#include "Geometry/Vector/interface/GlobalPoint.h"
#include <vector>


/** \class CaloCellGeometry
    
$Date: 2005/10/03 22:33:49 $
$Revision: 1.2 $
\author J. Mans, P. Meridiani
*/
class CaloCellGeometry {
public:
  CaloCellGeometry() {;} ;
  virtual ~CaloCellGeometry() {;};
  virtual bool inside(const GlobalPoint & point) const =0 ;  
  virtual const std::vector<GlobalPoint> & getCorners() const =0 ;
  
  /** return the position of reference for this cell */
  const GlobalPoint& getPosition() const { return refPoint_; }
  void setPosition(const GlobalPoint& refPoint) { refPoint_=refPoint; }

protected:
  CaloCellGeometry(const GlobalPoint& positionOfReference);
private:
  GlobalPoint refPoint_;
};

#endif
