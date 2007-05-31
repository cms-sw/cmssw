#include "RecoTracker/NuclearSeedGenerator/interface/TangentCircle.h"

#define PI 3.1415926

TangentCircle::TangentCircle(const GlobalVector& direction, const GlobalPoint& inner, const GlobalPoint& outer) : 
       theInnerPoint(inner), theOuterPoint(outer), theVertexPoint(inner) {
   double x1 = inner.x();
   double y1 = inner.y();
   double x2 = outer.x();
   double y2 = outer.y();
   double alpha1 = (direction.y() != 0) ? atan(-direction.x()/direction.y()) : PI/2 ;
   double denominator = 2*((x1-x2)*cos(alpha1)+(y1-y2)*sin(alpha1));
   theRho = (denominator != 0) ? ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))/denominator : 1E12;

   // variable not calculated
   theX0 = 1000;
   theY0 = 1000;
}

TangentCircle::TangentCircle(const GlobalPoint& outerPoint, const GlobalPoint& innerPoint, const GlobalPoint& vertexPoint) : 
     theInnerPoint(theInnerPoint), theOuterPoint(outerPoint), theVertexPoint(vertexPoint) {
     FastCircle circle(outerPoint, innerPoint, vertexPoint);
     theX0 = circle.x0();
     theY0 = circle.y0();
     theRho = circle.rho();
}

TangentCircle::TangentCircle(const TangentCircle& primCircle, const GlobalPoint& outerPoint, const GlobalPoint& innerPoint) {

   // Initial vertex used = outerPoint of the primary circle (should be the first estimation of the nuclear interaction position)
   GlobalPoint vertex = primCircle.outerPoint();
   
   // get the circle which pass through outerPoint, innerPoint and teh vertex
   TangentCircle secCircle( outerPoint, innerPoint, vertex );

   double minCond = isTangent(primCircle, secCircle);
   
   theInnerPoint = innerPoint;
   theOuterPoint = outerPoint;
   theX0 = secCircle.x0();
   theY0 = secCircle.y0();
   theRho = secCircle.rho();
}

double TangentCircle::isTangent(const TangentCircle& primCircle, const TangentCircle& secCircle) const {
   // return a value that should be equal to 0 if primCircle and secCircle are tangent
   return (primCircle.rho() - secCircle.rho())*(primCircle.rho() - secCircle.rho()) 
           - (primCircle.x0() - secCircle.x0())*(primCircle.x0() - secCircle.x0())
           - (primCircle.y0() - secCircle.y0())*(primCircle.y0() - secCircle.y0());
}

GlobalVector TangentCircle::direction(const GlobalPoint& point) const {
     GlobalVector dir(point.y() - theY0, point.x() - theX0, point.z());
     dir/=dir.mag();
     return dir;
}
