#include "RecoTracker/NuclearSeedGenerator/interface/TangentCircle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#define PI 3.1415926

// TODO: is not valid don't do any calculations and return init values
TangentCircle::TangentCircle(const GlobalVector& direction, const GlobalPoint& inner, const GlobalPoint& outer) : 
       theInnerPoint(inner), theOuterPoint(outer), theVertexPoint(inner) {

   if(theInnerPoint.perp2() > theOuterPoint.perp2()) { valid = false; }
   else valid=true;

   double x1 = inner.x();
   double y1 = inner.y();
   double x2 = outer.x();
   double y2 = outer.y();
   double alpha1 = (direction.y() != 0) ? atan(-direction.x()/direction.y()) : PI/2 ;
   double denominator = 2*((x1-x2)*cos(alpha1)+(y1-y2)*sin(alpha1));
   theRho = (denominator != 0) ? ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))/denominator : 1E12;

   // TODO : variable not yet calculated look in nucl.C
   theX0 = 1E10;
   theY0 = 1E10;

   theDirectionAtVertex = direction;
   theDirectionAtVertex/=theDirectionAtVertex.mag();

   //theCharge = (theRho>0) ? -1 : 1;

   theCharge = 0;
   theRho = fabs(theRho);

   theVertexError = (theInnerPoint-theOuterPoint).mag();
}

TangentCircle::TangentCircle(const GlobalPoint& outerPoint, const GlobalPoint& innerPoint, const GlobalPoint& vertexPoint) : 
     theInnerPoint(innerPoint), theOuterPoint(outerPoint), theVertexPoint(vertexPoint) {
     FastCircle circle(outerPoint, innerPoint, vertexPoint);
     theX0 = circle.x0();
     theY0 = circle.y0();
     theRho = circle.rho();
     theVertexError = 0;
     theCharge = 0;
     theDirectionAtVertex = GlobalVector(1000, 1000, 1000);
     if(theInnerPoint.perp2() > theOuterPoint.perp2() || !circle.isValid()) { valid = false; }
     else valid=true;
}

TangentCircle::TangentCircle(const TangentCircle& primCircle, const GlobalPoint& outerPoint, const GlobalPoint& innerPoint) {

   if(theInnerPoint.perp2() > theOuterPoint.perp2()) { valid = false; }
   else valid = true;

   int NITER = 10; 

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
   int dir = 1;
   double theta = deltaTheta/(NITER-1);

   for(int i=0; i<NITER; i++) { 
   
     // get the circle which pass through outerPoint, innerPoint and the vertex
     TangentCircle secCircle( SecOuterPoint, SecInnerPoint, vertex );

     // get a value relative to the tangentness of the 2 circles
     double minCond = isTangent(primCircle, secCircle);

     // double dirDiff = (primCircle.direction(vertex) - secCircle.direction(vertex)).mag();
     // if( dirDiff > 1) dirDiff = 2-dirDiff;

     if(minCond < minTangentCondition) { 
                minTangentCondition = minCond;
                theCorrectSecCircle = secCircle;
                vertex = getPosition( primCircle, secCircle.vertexPoint(), theta, dir );
                if( i==0 && ((vertex-SecInnerPoint).mag() > (InitialVertex-SecInnerPoint).mag()) ) {
                       dir=-1;
                       vertex = getPosition( primCircle, InitialVertex, theta, dir );
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
   theCharge = 0;
   theDirectionAtVertex = GlobalVector(1000, 1000, 1000);

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

     if(theY0 > 1E9 || theX0 > 1E9) {
        LogDebug("NuclearSeedGenerator") << "Center of TangentCircle not calculated but used !!!" << "\n";
     }

     // calculate the direction perpendicular to the vector v = point - center_of_circle
     GlobalVector dir(point.y() - theY0, theX0 - point.x(), 0);

     dir/=dir.mag();

     // Check the sign :
     GlobalVector fastDir = theOuterPoint - theInnerPoint;
     double diff = (dir - fastDir).mag();
     double sum = (dir + fastDir).mag();

     if( sum < diff ) dir = (-1)*dir;

     return dir;
}

GlobalVector TangentCircle::directionAtVertex()  {
      if(theDirectionAtVertex.x() > 999) 
                theDirectionAtVertex = direction(theVertexPoint);
      return theDirectionAtVertex;
}

GlobalPoint TangentCircle::getPosition(const TangentCircle& circle, const GlobalPoint& initalPosition, double theta, int dir) const {
             
            int sign[3];
            double x2 = initalPosition.x();
            double y2 = initalPosition.y();

            if( (x2>circle.x0()) && dir >0) { sign[0] = 1;  sign[1] = -1; sign[2] = -1; }
            if( (x2>circle.x0()) && dir <0) { sign[0] = 1;  sign[1] = 1; sign[2] = 1; }
            if( (x2<circle.x0()) && dir >0) { sign[0] = -1;  sign[1] = 1; sign[2] = -1; }
            if( (x2<circle.x0()) && dir <0) { sign[0] = -1;  sign[1] = -1; sign[2] = 1; }

            double l = 2*circle.rho()*sin(theta/2);
            double alpha = atan((y2-circle.y0())/(x2-circle.x0()));
            double beta = PI/2-theta/2;
            double gamma = PI + sign[2]* alpha - beta;

            double xnew = x2 + sign[0]*l*cos(gamma);
            double ynew = y2 + sign[1]*l*sin(gamma);

            return GlobalPoint( xnew, ynew, 0 );
}

double TangentCircle::curvatureError() {
   if( (theInnerPoint - theVertexPoint).mag() < theVertexError ) {
        TangentCircle circle1( directionAtVertex() , theVertexPoint - theVertexError*directionAtVertex(), theOuterPoint);
        TangentCircle circle2( directionAtVertex() , theVertexPoint + theVertexError*directionAtVertex(), theOuterPoint);
        return fabs(1/circle1.rho() - 1/circle2.rho());
   }
   else {
       TangentCircle circle1( theOuterPoint, theInnerPoint, theVertexPoint - theVertexError*directionAtVertex());
       TangentCircle circle2( theOuterPoint, theInnerPoint, theVertexPoint + theVertexError*directionAtVertex());
       return fabs(1/circle1.rho() - 1/circle2.rho());
   }
}

int TangentCircle::charge(float magz) {

   if(theCharge != 0) return theCharge;

   if(theX0 > 1E9 || theY0 > 1E9) theCharge = chargeLocally(magz, directionAtVertex());
   else {
       GlobalPoint  center(theX0, theY0, 0);
       GlobalVector u = center - theVertexPoint;
       GlobalVector v = directionAtVertex();

       // F = force vector
       GlobalVector F( v.y() * magz, -v.x() * magz, 0);
       if( u.x() * F.x() + u.y() * F.y() > 0) theCharge=-1;
       else theCharge=1;

       if(theCharge != chargeLocally(magz, v)) {
          LogDebug("NuclearSeedGenerator") << "Inconsistency in calculation of the charge" << "\n";
     }

   }
   return theCharge;
}

int TangentCircle::chargeLocally(float magz, GlobalVector v) const {
    GlobalVector  u = theOuterPoint - theVertexPoint;
    double tz = v.x() * u.y() - v.y() * u.x() ;

    if(tz * magz > 0) return 1; else return -1;
}
