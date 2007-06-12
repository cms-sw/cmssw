#include "RecoTracker/NuclearSeedGenerator/interface/TangentCircle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

   theVertexError = (theInnerPoint-theOuterPoint).mag()/2;
   // TODO: add charge (maybe in SeedFromNuclearInteraction) -> look example in FastHelix!!!!
}

TangentCircle::TangentCircle(const GlobalPoint& outerPoint, const GlobalPoint& innerPoint, const GlobalPoint& vertexPoint) : 
     theInnerPoint(innerPoint), theOuterPoint(outerPoint), theVertexPoint(vertexPoint) {
     FastCircle circle(outerPoint, innerPoint, vertexPoint);
     theX0 = circle.x0();
     theY0 = circle.y0();
     theRho = circle.rho();
     theVertexError = 0;
}

TangentCircle::TangentCircle(const TangentCircle& primCircle, const GlobalPoint& outerPoint, const GlobalPoint& innerPoint) {

   int NITER = 5; 

   // Initial vertex used = outerPoint of the primary circle (should be the first estimation of the nuclear interaction position)
   GlobalPoint InitialVertex( primCircle.outerPoint().x() , primCircle.outerPoint().y(), 0);
   GlobalPoint SecInnerPoint( innerPoint.x(), innerPoint.y(), 0);
   GlobalPoint SecOuterPoint( outerPoint.x(), outerPoint.y(), 0);

   // distance between the initial vertex and the inner point of the secondary circle
   double s = (SecInnerPoint - InitialVertex).mag();
   double deltaTheta = s/primCircle.rho();

   double minTangentCondition = 1E12;
   TangentCircle theCorrectSecCircle;
   GlobalPoint vertex = InitialVertex;
   int direction = 1;
   double theta = deltaTheta/(NITER-1);

   for(int i=0; i<NITER; i++) { 
   
     // get the circle which pass through outerPoint, innerPoint and the vertex
     TangentCircle secCircle( SecOuterPoint, SecInnerPoint, vertex );

     // get a value relative to the tangentness of the 2 circles
     double minCond = isTangent(primCircle, secCircle);

     // double dirDiff = (primCircle.direction(vertex) - secCircle.direction(vertex)).mag();
     // if( dirDiff > 1) dirDiff = 2-dirDiff;

     LogDebug("NuclearSeedGenerator") << "Vertex position : " << vertex.x() << "  " << vertex.y() << "\n"
                                      << "isTangent condition: " << minCond << "\n";

     if(minCond < minTangentCondition) { 
                minTangentCondition = minCond;
                theCorrectSecCircle = secCircle;
                vertex = getPosition( primCircle, secCircle.vertexPoint(), theta, direction );
                if( i==0 && ((vertex-SecInnerPoint).mag() > (InitialVertex-SecInnerPoint).mag()) ) {
                       direction=-1;
                       vertex = getPosition( primCircle, InitialVertex, theta, direction );
                       LogDebug("NuclearSeedGenerator") << "Change direction to look for vertex" << "\n";
                 }
     }
     else break;
   
   }
   theInnerPoint = theCorrectSecCircle.innerPoint();
   theOuterPoint = theCorrectSecCircle.outerPoint();
   theVertexPoint = theCorrectSecCircle.vertexPoint();
   theX0 = theCorrectSecCircle.x0();
   theY0 = theCorrectSecCircle.y0();
   theRho = theCorrectSecCircle.rho();  

   theVertexError = s/NITER;
}

double TangentCircle::isTangent(const TangentCircle& primCircle, const TangentCircle& secCircle) const {
   // return a value that should be equal to 0 if primCircle and secCircle are tangent

   double distanceBetweenCircle = (primCircle.x0() - secCircle.x0())*(primCircle.x0() - secCircle.x0())
           + (primCircle.y0() - secCircle.y0())*(primCircle.y0() - secCircle.y0());
   double RadiusSum = (primCircle.rho() + secCircle.rho())*(primCircle.rho() + secCircle.rho());
   double RadiusDifference = (primCircle.rho() - secCircle.rho())*(primCircle.rho() - secCircle.rho());

   return std::min( fabs(RadiusSum-distanceBetweenCircle), fabs(RadiusDifference-distanceBetweenCircle) ); 
}

GlobalVector TangentCircle::direction(const GlobalPoint& point) const {
     GlobalVector dir(point.y() - theY0, point.x() - theX0, 0);
     dir/=dir.mag();
     return dir;
}

GlobalVector TangentCircle::directionAtVertex() const {
      //TODO : check if the sens is correct (for propagation)
      return direction(theVertexPoint);
}

GlobalPoint TangentCircle::getPosition(const TangentCircle& circle, const GlobalPoint& initalPosition, double theta, int direction) const {
             
            int sign[3];
            double x2 = initalPosition.x();
            double y2 = initalPosition.y();

            if( (x2>circle.x0()) && direction >0) { sign[0] = 1;  sign[1] = -1; sign[2] = -1; }
            if( (x2>circle.x0()) && direction <0) { sign[0] = 1;  sign[1] = 1; sign[2] = 1; }
            if( (x2<circle.x0()) && direction >0) { sign[0] = -1;  sign[1] = 1; sign[2] = -1; }
            if( (x2<circle.x0()) && direction <0) { sign[0] = -1;  sign[1] = -1; sign[2] = 1; }

            double l = 2*circle.rho()*sin(theta/2);
            double alpha = atan((y2-circle.y0())/(x2-circle.x0()));
            double beta = PI/2-theta/2;
            double gamma = PI + sign[2]* alpha - beta;

            double xnew = x2 + sign[0]*l*cos(gamma);
            double ynew = y2 + sign[1]*l*sin(gamma);

            return GlobalPoint( xnew, ynew, 0 );
}

double TangentCircle::rhoError() const {
   if( (theInnerPoint - theVertexPoint).mag() < theVertexError ) {
        TangentCircle circle1( directionAtVertex() , theVertexPoint - theVertexError*directionAtVertex(), theOuterPoint);
        TangentCircle circle2( directionAtVertex() , theVertexPoint + theVertexError*directionAtVertex(), theOuterPoint);
        return fabs(circle1.rho() - circle2.rho());
   }
   else {
       TangentCircle circle1( theOuterPoint, theInnerPoint, theVertexPoint - theVertexError*directionAtVertex());
       TangentCircle circle2( theOuterPoint, theInnerPoint, theVertexPoint + theVertexError*directionAtVertex());
       return fabs(circle1.rho() - circle2.rho());
   }
}
       
