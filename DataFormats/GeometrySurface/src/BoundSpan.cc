#include "DataFormats/GeometrySurface/interface/BoundSpan.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"

namespace boundSpan {
  
  std::pair<float, float> computePhiSpan( const BoundSurface& plane) {
    typedef std::pair<float, float> return_type;  
    const TrapezoidalPlaneBounds* trapezoidalBounds( dynamic_cast<const TrapezoidalPlaneBounds*>(&(plane.bounds())));
    const RectangularPlaneBounds* rectangularBounds( dynamic_cast<const RectangularPlaneBounds*>(&(plane.bounds())));  
    
    float corners[4];
    if (trapezoidalBounds) {
      std::vector<float> const & parameters = (*trapezoidalBounds).parameters();
      
      float hbotedge = parameters[0];
      float htopedge = parameters[1];
      float hapothem = parameters[3];   
      
      corners[0] = plane.toGlobal( LocalPoint( -htopedge, hapothem, 0.)).barePhi();
      corners[1] = plane.toGlobal( LocalPoint(  htopedge, hapothem, 0.)).barePhi();
      corners[2] = plane.toGlobal( LocalPoint(  hbotedge, -hapothem, 0.)).barePhi();
      corners[3] = plane.toGlobal( LocalPoint( -hbotedge, -hapothem, 0.)).barePhi();
      
    }else if(rectangularBounds) {
      float length = rectangularBounds->length();
      float width  = rectangularBounds->width();   
      
      corners[0] = plane.toGlobal( LocalPoint( -width/2, -length/2, 0.)).barePhi();
      corners[1] = plane.toGlobal( LocalPoint( -width/2, +length/2, 0.)).barePhi();
      corners[2] = plane.toGlobal( LocalPoint( +width/2, -length/2, 0.)).barePhi();
      corners[3] = plane.toGlobal( LocalPoint( +width/2, +length/2, 0.)).barePhi();
    }else{  
      return return_type(-Geom::pi(),Geom::pi()); 
    }
    
    float phimin = corners[0];
    float phimax = phimin;
    for ( int i = 1; i < 4; i++ ) {
      float cPhi = corners[i];
      if ( Geom::phiLess( cPhi, phimin)) { phimin = cPhi; }
      if ( Geom::phiLess( phimax, cPhi)) { phimax = cPhi; }
    }
    return return_type( phimin, phimax);
    
  }
  
  
}
