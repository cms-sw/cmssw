// $Id: Thrust.cc,v 1.15 2011/10/12 15:30:11 gowdy Exp $
#include "PhysicsTools/CandUtils/interface/Thrust.h"
#include <cmath>
using namespace reco;
const double pi = M_PI, pi2 = 2 * pi, pi_2 = pi / 2, pi_4 = pi / 4;

void Thrust::init(const std::vector<const Candidate*>& cands) {
  int i = 0;
  for(std::vector<const Candidate*>::const_iterator t = cands.begin(); 
      t != cands.end(); ++t, ++i)
    pSum_ += (p_[i] = (*t)->momentum()).r();
  axis_ = axis(finalAxis(initialAxis()));
  if (axis_.z() < 0) axis_ *= -1;
  thrust_ = thrust(axis_);
}

Thrust::ThetaPhi Thrust::initialAxis() const {
  static const int nSegsTheta = 10, nSegsPhi = 10, nSegs = nSegsTheta * nSegsPhi;
  int i, j;
  double thr[nSegs], max = 0;
  int indI = 0, indJ = 0, index = -1;
  for (i = 0; i < nSegsTheta ; ++i) {
    double z = cos(pi * i / (nSegsTheta - 1));
    double r = sqrt(1 - z*z);
    for (j = 0; j < nSegsPhi ; ++j) {
      double phi = pi2 * j / nSegsPhi;
      thr[i * nSegsPhi + j] = thrust(Vector(r*cos(phi), r*sin(phi), z));
      if (thr[i*nSegsPhi + j] > max) {
        index = i*nSegsPhi + j;
        indI = i; indJ = j;
        max = thr[index];
      }
    }
  }

  // take max and one point on either size, fitting to a parabola and
  // extrapolating to the real max.  Do this separately for each dimension.
  // y = a x^2 + b x + c.  At the max, x = 0, on either side, x = +/-1.
  // do phi first
  double a, b, c = max;
  int ind1 = indJ + 1;
  if (ind1 >= nSegsPhi) ind1 -= nSegsPhi;
  int ind2 = indJ - 1;
  if (ind2 < 0) ind2 += nSegsPhi;
  a = (thr[ind1] + thr[ind2] - 2*c) / 2;
  b = thr[ind1] - a - c;
  double maxPhiInd = 0;
  if (a != 0) maxPhiInd = -b/(2*a);
  double maxThetaInd;
  if (indI == 0 || indI == (nSegsTheta - 1)) 
    maxThetaInd = indI;
  else {
    ind1 = indI + 1;
    ind2 = indI - 1;
    a = (thr[ind1] + thr[ind2] - 2*c) / 2;
    b = thr[ind1] - a - c; 
    maxThetaInd = 0;
    if (a != 0) maxThetaInd = - b/(2*a);
  }
  return ThetaPhi(pi*(maxThetaInd + indI) / (nSegsTheta - 1),
		  pi2*(maxPhiInd + indJ) / nSegsPhi);
}

Thrust::ThetaPhi Thrust::finalAxis(ThetaPhi best) const {
  static const double epsilon = 0.0001;
  double maxChange1=0.0, maxChange2=0.0, a=0.0, b=0.0, c=0.0, thr=0.0;
  int mandCt = 3, maxCt = 1000;
  bool done;
  do { 
    parabola(a, b, c, 
	     axis(best), 
	     axis(best.theta + epsilon, best.phi), 
	     axis(best.theta - epsilon, best.phi));
    maxChange1 = 10*(b < 0 ? -1 : 1);
    if (a != 0) maxChange1 = - b/(2*a);
    while (fabs(maxChange1 * epsilon) > pi_4) maxChange1 /= 2;
    if (maxChange1 == 0 && (best.theta == 0 || best.theta == pi)) { 
      best.phi += pi_2;
      if (best.phi > pi2) best.phi -= pi2;
      parabola(a, b, c, 
		axis(best),
		axis(best.theta + epsilon, best.phi),
		axis(best.theta - epsilon, best.phi));
      maxChange1 = 10 * (b < 0 ? -1 : 1);
      if (a != 0) maxChange1 = - b / (2 * a);
    }
    do {
      // L.L.: fixed odd behavoir adding epsilon (???)
      thr = thrust(axis(best.theta + maxChange1 * epsilon, best.phi)) + epsilon;
      if (thr < c) maxChange1 /= 2;
    } while (thr < c);

    best.theta += maxChange1 * epsilon;
    if (best.theta > pi) {
      best.theta = pi - (best.theta - pi);
      best.phi += pi;
      if (best.phi > pi2) best.phi -= pi2;
    }
    if (best.theta < 0) {
      best.theta *= -1;
      best.phi += pi;
      if (best.phi > pi2) best.phi -= pi2;
    }
    parabola(a, b, c, 
	      axis(best),
	      axis(best.theta, best.phi + epsilon),
	      axis(best.theta, best.phi - epsilon));
    maxChange2 = 10 * (b < 0 ? -1 : 1);
    if (a != 0) maxChange2 = - b / (2 * a);
    while (fabs(maxChange2 * epsilon) > pi_4) { maxChange2 /= 2; }
    do {
      // L.L.: fixed odd behavoir adding epsilon
      thr = thrust(axis(best.theta, best.phi + maxChange2 * epsilon)) + epsilon;
      if (thr < c) maxChange2 /= 2;
    } while (thr < c);
    best.phi += maxChange2 * epsilon;
    if (best.phi > pi2) best.phi -= pi2;
    if (best.phi < 0) best.phi += pi2;
    if (mandCt > 0) mandCt --;
    maxCt --;
    done = (fabs(maxChange1) > 1 || fabs(maxChange2) > 1 || mandCt) && (maxCt > 0);
  } while (done);

  return best;
}

void Thrust::parabola(double & a, double & b, double & c, 
		       const Vector & a1, const Vector & a2, const Vector & a3) const {
  double t1 = thrust(a1), t2 = thrust(a2), t3 = thrust(a3);
  a = (t2 - 2 * c + t3) / 2;
  b = t2 - a - c;
  c = t1;
}

Thrust::Vector Thrust::axis(double theta, double phi) const {
  double theSin = sin(theta);
  return Vector(theSin * cos(phi), theSin * sin(phi), cos(theta));
}

double Thrust::thrust(const Vector & axis) const {
  double result = 0;
  double sum = 0;
  for (unsigned int i = 0; i < n_; ++i)
    sum += fabs(axis.Dot(p_[i]));
  if (pSum_ > 0) result = sum / pSum_;
  return result;
}
