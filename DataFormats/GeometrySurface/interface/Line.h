#ifndef Geom_Line_H
#define Geom_Line_H

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

/** A line in 3D space.
 */

class Line {
public:
  typedef GlobalPoint PositionType;
  typedef GlobalVector DirectionType;

  Line() {}

  //Line( const PositionType& pos, const  DirectionType& dir) :
  Line(PositionType& pos, DirectionType& dir) : thePos(pos), theDir(dir.unit()) {}

  ~Line(){};

  //const PositionType& position()   const { return thePos;}
  //const DirectionType& direction() const { return theDir;}
  PositionType position() const { return thePos; }
  DirectionType direction() const { return theDir; }

  GlobalPoint closerPointToLine(const Line& aLine) const {
    GlobalPoint V = aLine.position();
    GlobalVector J = aLine.direction();
    GlobalVector Q = theDir - J.dot(theDir) * J;
    double lambda = Q.dot(V - thePos) / Q.dot(theDir);
    return thePos + lambda * theDir;
  }

  GlobalVector distance(const Line& aLine) const {
    GlobalPoint V = aLine.position();
    GlobalVector J = aLine.direction();
    GlobalVector P = (theDir.cross(J)).unit();
    GlobalVector D;
    D = P.dot(thePos - V) * P;
    return D;
  }

  GlobalVector distance(const GlobalPoint& aPoint) const {
    GlobalVector P(aPoint.x(), aPoint.y(), aPoint.z());
    GlobalVector T0(thePos.x(), thePos.y(), thePos.z());
    return T0 - P + theDir.dot(P - T0) * theDir;
  }

private:
  PositionType thePos;
  DirectionType theDir;
};

#endif  // Geom_Line_H
