#include "FastSimulation/SimplifiedGeometryPropagator/interface/StraightTrajectory.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Constants.h"

#include "FastSimulation/SimplifiedGeometryPropagator/interface/SimplifiedGeometry.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/BarrelSimplifiedGeometry.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/ForwardSimplifiedGeometry.h"

double fastsim::StraightTrajectory::nextCrossingTimeC(const fastsim::BarrelSimplifiedGeometry& layer,
                                                      bool onLayer) const {
  //
  // solve the equation
  // (x^2 + y^2 = R^2),   (1)
  // x = x_0 + v_x*t,     (2)
  // y = y_0 + v_y*t      (3)
  // for t
  //
  // subsitute (2) and (3) in (1):
  // => t^2*(v_x^2 + v_y^2) + t*( 2*x_0*v_x + 2*y_0*v_y ) + x_0^2 + y_0^2 - R^2 = 0
  // => t = (-b +/- sqrt(b^2 - ac) ) / a       see https://en.wikipedia.org/wiki/Quadratic_formula, divide equation by 2
  // with a = v_x^2 + v_y^2;
  // with b = x_0*v_x + y_0*v_y
  // with c = x_0^2 + y_0^2 - R^2
  //
  // substitute: v_x = p_x / E * c  ( note: extra * c absorbed in p_x units)
  // substitute: v_y = p_y / E * c  ( note: extra * c absorbed in p_y units)
  // => t*c = (-b +/- sqrt(b^2 - ac) ) / a * E
  // with a = p_x^2 + p_y^2
  // with b = p_x*x_0 + p_y*y_0
  // with c = x_0^2 + y_0^2 - R^2
  double a = momentum_.Perp2();
  double b = (position_.X() * momentum_.X() + position_.Y() * momentum_.Y());
  double c = position_.Perp2() - layer.getRadius() * layer.getRadius();

  double delta = b * b - a * c;
  if (delta < 0) {
    return -1;
  }
  double sqrtDelta = sqrt(delta);

  //
  // return the earliest solution > 0,
  // return -1 if no positive solution exists
  // note: 'a' always positive, sqrtDelta always positive
  //
  double tc1 = (-b - sqrtDelta) / a * momentum_.E();
  double tc2 = (-b + sqrtDelta) / a * momentum_.E();

  if (onLayer && tc2 > 0) {
    bool particleMovesInwards = momentum_.X() * position_.X() + momentum_.Y() * position_.Y() < 0;

    double posX2 = position_.X() + momentum_.X() / momentum_.E() * tc2;
    double posY2 = position_.Y() + momentum_.Y() / momentum_.E() * tc2;
    bool particleMovesInwards2 = momentum_.X() * posX2 + momentum_.Y() * posY2 < 0;

    if (particleMovesInwards != particleMovesInwards2) {
      return tc2;
    }

    return -1;
  }

  if (tc1 > 0) {
    return tc1;
  } else if (tc2 > 0) {
    return tc2;
  }

  return -1.;
}

void fastsim::StraightTrajectory::move(double deltaTimeC) {
  position_.SetXYZT(position_.X() + momentum_.X() / momentum_.E() * deltaTimeC,
                    position_.Y() + momentum_.Y() / momentum_.E() * deltaTimeC,
                    position_.Z() + momentum_.Z() / momentum_.E() * deltaTimeC,
                    position_.T() + deltaTimeC / fastsim::Constants::speedOfLight);
}
