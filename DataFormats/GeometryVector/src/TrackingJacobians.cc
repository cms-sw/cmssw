#include "DataFormats/GeometryVector/interface/TrackingJacobians.h"

// Code moved from TrackingTools/AnalyticalJacobians

AlgebraicMatrix65 jacobianCurvilinearToCartesian(const GlobalVector& momentum, int q) {
  AlgebraicMatrix65 theJacobian;
  GlobalVector xt = momentum;
  GlobalVector yt(-xt.y(), xt.x(), 0.);
  GlobalVector zt = xt.cross(yt);

  const GlobalVector& pvec = momentum;
  double pt = pvec.perp();
  // for neutrals: qbp is 1/p instead of q/p -
  //   equivalent to charge 1
  if (q == 0)
    q = 1;

  xt = xt.unit();
  if (fabs(pt) > 0) {
    yt = yt.unit();
    zt = zt.unit();
  }

  AlgebraicMatrix66 R;
  R(0, 0) = xt.x();
  R(0, 1) = yt.x();
  R(0, 2) = zt.x();
  R(1, 0) = xt.y();
  R(1, 1) = yt.y();
  R(1, 2) = zt.y();
  R(2, 0) = xt.z();
  R(2, 1) = yt.z();
  R(2, 2) = zt.z();
  R(3, 3) = 1.;
  R(4, 4) = 1.;
  R(5, 5) = 1.;

  double p = pvec.mag(), p2 = p * p;
  double sinlambda = pvec.z() / p, coslambda = pt / p;
  double sinphi = pvec.y() / pt, cosphi = pvec.x() / pt;

  theJacobian(1, 3) = 1.;
  theJacobian(2, 4) = 1.;
  theJacobian(3, 0) = -q * p2 * coslambda * cosphi;
  theJacobian(3, 1) = -p * sinlambda * cosphi;
  theJacobian(3, 2) = -p * coslambda * sinphi;
  theJacobian(4, 0) = -q * p2 * coslambda * sinphi;
  theJacobian(4, 1) = -p * sinlambda * sinphi;
  theJacobian(4, 2) = p * coslambda * cosphi;
  theJacobian(5, 0) = -q * p2 * sinlambda;
  theJacobian(5, 1) = p * coslambda;
  theJacobian(5, 2) = 0.;

  //ErrorPropagation:
  //    C(Cart) = R(6*6) * J(6*5) * C(Curvi) * J(5*6)_T * R(6*6)_T
  theJacobian = R * theJacobian;
  //dbg::dbg_trace(1,"Cu2Ca", globalParameters.vector(),theJacobian);
  return theJacobian;
}

AlgebraicMatrix56 jacobianCartesianToCurvilinear(const GlobalVector& momentum, int q) {
  AlgebraicMatrix56 theJacobian;
  GlobalVector xt = momentum;
  GlobalVector yt(-xt.y(), xt.x(), 0.);
  GlobalVector zt = xt.cross(yt);
  const GlobalVector& pvec = momentum;
  double pt = pvec.perp(), p = pvec.mag();
  double px = pvec.x(), py = pvec.y(), pz = pvec.z();
  double pt2 = pt * pt, p2 = p * p, p3 = p * p * p;
  // for neutrals: qbp is 1/p instead of q/p -
  //   equivalent to charge 1
  if (q == 0)
    q = 1;
  xt = xt.unit();
  if (fabs(pt) > 0) {
    yt = yt.unit();
    zt = zt.unit();
  }

  AlgebraicMatrix66 R;
  R(0, 0) = xt.x();
  R(0, 1) = xt.y();
  R(0, 2) = xt.z();
  R(1, 0) = yt.x();
  R(1, 1) = yt.y();
  R(1, 2) = yt.z();
  R(2, 0) = zt.x();
  R(2, 1) = zt.y();
  R(2, 2) = zt.z();
  R(3, 3) = 1.;
  R(4, 4) = 1.;
  R(5, 5) = 1.;

  theJacobian(0, 3) = -q * px / p3;
  theJacobian(0, 4) = -q * py / p3;
  theJacobian(0, 5) = -q * pz / p3;
  if (fabs(pt) > 0) {
    //theJacobian(1,3) = (px*pz)/(pt*p2); theJacobian(1,4) = (py*pz)/(pt*p2); theJacobian(1,5) = -pt/p2; //wrong sign
    theJacobian(1, 3) = -(px * pz) / (pt * p2);
    theJacobian(1, 4) = -(py * pz) / (pt * p2);
    theJacobian(1, 5) = pt / p2;
    theJacobian(2, 3) = -py / pt2;
    theJacobian(2, 4) = px / pt2;
    theJacobian(2, 5) = 0.;
  }
  theJacobian(3, 1) = 1.;
  theJacobian(4, 2) = 1.;
  theJacobian = theJacobian * R;
  //dbg::dbg_trace(1,"Ca2Cu", globalParameters.vector(),theJacobian);
  return theJacobian;
}
