#ifndef GEOMETRY_CALOGEOMETRY_CALOCELLGEOMETRY_H
#define GEOMETRY_CALOGEOMETRY_CALOCELLGEOMETRY_H 1

#include "Geometry/Vector/interface/GlobalPoint.h"
#include <vector>

namespace cms {

  /** \class CaloCellGeometry
      
  $Date: $
  $Revision: $
  \author J. Mans, P. Meridiani
  */
  class CaloCellGeometry {
  public:
    virtual ~CaloCellGeometry() {;};
    virtual bool inside(const GlobalPoint & point) const =0 ;  
    virtual const std::vector<GlobalPoint> & getCorners() const =0 ;
   
    /** return the position of reference for this cell */
    const GlobalPoint& getPosition() const { return refPoint_; }
  protected:
    CaloCellGeometry(const GlobalPoint& positionOfReference);
  private:
    GlobalPoint refPoint_;
  };

}

#endif
