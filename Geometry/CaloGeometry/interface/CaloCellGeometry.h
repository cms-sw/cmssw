#ifndef GEOMETRY_CALOGEOMETRY_CALOCELLGEOMETRY_H
#define GEOMETRY_CALOGEOMETRY_CALOCELLGEOMETRY_H 1

#include "Geometry/Vector/interface/GlobalPoint.h"
#include <vector>


/** \class CaloCellGeometry

  Abstract base class for an individual cell's geometry.
    
$Date: 2005/10/27 17:46:06 $
$Revision: 1.3 $
\author J. Mans, P. Meridiani
*/
class CaloCellGeometry {
public:
  CaloCellGeometry() {;} ;
  virtual ~CaloCellGeometry() {;};
  /// Returns true if the specified point is inside this cell
  virtual bool inside(const GlobalPoint & point) const =0 ;  
  /// Returns the corner points of this cell's volume
  virtual const std::vector<GlobalPoint> & getCorners() const =0 ;
  
  /// Returns the position of reference for this cell 
  const GlobalPoint& getPosition() const { return refPoint_; }
  /// Setter function for the position of reference
  void setPosition(const GlobalPoint& refPoint) { refPoint_=refPoint; }

protected:
  CaloCellGeometry(const GlobalPoint& positionOfReference);
private:
  GlobalPoint refPoint_;
};

#endif
