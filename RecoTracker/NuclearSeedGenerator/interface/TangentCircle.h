#ifndef _TangentCircle_H_
#define _TangentCircle_H_

#include "RecoTracker/TkSeedGenerator/interface/FastCircle.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

class TangentCircle
{

 public :
     TangentCircle(){}

     /// Calculate the circle from 2 points on the circle (the vertex=innerPoint and the outerPoint)
     /// and the tangent direction at the inner point
     TangentCircle(const GlobalVector& direction, const GlobalPoint& innerPoint, const GlobalPoint& outerPoint); 

     /// Copy of FastCircle
     TangentCircle(const GlobalPoint& outerPoint, const GlobalPoint& innerPoint, const GlobalPoint& vertexPoint); 

     /// Calculate the parameters of a circle which pass by 2 points (innerPoint and outerPoint) and which is tangent to primCircle 
     TangentCircle(const TangentCircle& primCircle, const GlobalPoint& outerPoint, const GlobalPoint& innerPoint);

     GlobalVector direction(const GlobalPoint& point) const;

     /// Return the direction at the vertex
     GlobalVector directionAtVertex() const;

     double x0() const {return theX0;}
   
     double y0() const {return theY0;}
 
     double rho() const {return theRho;}

     GlobalPoint outerPoint() const { return theOuterPoint; }

     GlobalPoint innerPoint() const { return theInnerPoint; }

     GlobalPoint vertexPoint() const { return theVertexPoint; }

     double vertexError() const { return theVertexError; }

     double rhoError() const;

 private :
     GlobalPoint theInnerPoint;
     GlobalPoint theOuterPoint;
     GlobalPoint theVertexPoint;
     
     double theX0;  /**< x center of the circle             */
     double theY0;  /**< y center of the circle             */
     double theRho; /**< Signed radius of the circle (=q*R) */

     double theVertexError;  /**< the error on the vertex position along the direction of the circle at this point */

     double isTangent(const TangentCircle& primCircle, const TangentCircle& secCircle) const;
     GlobalPoint getPosition(const TangentCircle& circle, const GlobalPoint& initalPosition, double theta, int direction) const;
};

#endif
