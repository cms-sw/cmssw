#ifndef _TangentCircle_H_
#define _TangentCircle_H_

#include "RecoTracker/TkSeedGenerator/interface/FastCircle.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

class TangentCircle
{

 public :
     /// Calculate the circle from 2 points on the circle (the vertex=innerPoint and the outerPoint)
     /// and the tangent direction at the inner point
     TangentCircle(const GlobalVector& direction, const GlobalPoint& innerPoint, const GlobalPoint& outerPoint); 

     /// Calculate the parameters of a circle which pass by 2 points (innerPoint and outerPoint) and which is tangent to the circle (x0, y0, R) 
     /// Copy of FastCircle
     TangentCircle(const GlobalPoint& outerPoint, const GlobalPoint& innerPoint, const GlobalPoint& vertexPoint); 

     GlobalVector direction(const GlobalPoint& point) const;

     double x0() const {return theX0;}
   
     double y0() const {return theY0;}
 
     double rho() const {return theRho;}

 private :
     GlobalPoint theInnerPoint;
     GlobalPoint theOuterPoint;
     GlobalPoint theVertexPoint;
     
     double theX0;  /**< x center of the circle             */
     double theY0;  /**< y center of the circle             */
     double theRho; /**< Signed radius of the circle (=q*R) */
};

#endif
