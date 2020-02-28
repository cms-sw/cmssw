// -----------------------------------------------------------------------------
//  Prototype for a particle class
// -----------------------------------------------------------------------------
//  $Date: 2007/09/07 16:46:22 $
//  $Revision: 1.13 $
// -----------------------------------------------------------------------------
//  Author: Stephan Wynhoff - RWTH-Aachen (Email: Stephan.Wynhoff@cern.ch)
// -----------------------------------------------------------------------------
#include "CommonTools/BaseParticlePropagator/interface/RawParticle.h"

#include <cmath>

RawParticle::RawParticle(const XYZTLorentzVector& p) : myMomentum(p) {}

RawParticle::RawParticle(const int id, const XYZTLorentzVector& p, double mass, double charge)
    : myMomentum(p), myCharge{charge}, myMass{mass}, myId{id} {}

RawParticle::RawParticle(
    const int id, const XYZTLorentzVector& p, const XYZTLorentzVector& xStart, double mass, double charge)
    : myMomentum(p), myVertex{xStart}, myCharge{charge}, myMass{mass}, myId{id} {}

RawParticle::RawParticle(const XYZTLorentzVector& p, const XYZTLorentzVector& xStart, double charge)
    : myMomentum(p), myVertex{xStart}, myCharge{charge} {}

RawParticle::RawParticle(double px, double py, double pz, double e, double charge)
    : myMomentum(px, py, pz, e), myCharge{charge} {}

void RawParticle::setStatus(int istat) { myStatus = istat; }

void RawParticle::setMass(float m) { myMass = m; }

void RawParticle::setCharge(float q) { myCharge = q; }

void RawParticle::chargeConjugate() {
  myId = -myId;
  myCharge = -1 * myCharge;
}

void RawParticle::setT(const double t) { myVertex.SetE(t); }

void RawParticle::rotate(double angle, const XYZVector& raxis) {
  Rotation r(raxis, angle);
  XYZVector v(r * myMomentum.Vect());
  setMomentum(v.X(), v.Y(), v.Z(), E());
}

void RawParticle::rotateX(double rphi) {
  RotationX r(rphi);
  XYZVector v(r * myMomentum.Vect());
  setMomentum(v.X(), v.Y(), v.Z(), E());
}

void RawParticle::rotateY(double rphi) {
  RotationY r(rphi);
  XYZVector v(r * myMomentum.Vect());
  setMomentum(v.X(), v.Y(), v.Z(), E());
}

void RawParticle::rotateZ(double rphi) {
  RotationZ r(rphi);
  XYZVector v(r * myMomentum.Vect());
  setMomentum(v.X(), v.Y(), v.Z(), E());
}

void RawParticle::boost(double betax, double betay, double betaz) {
  Boost b(betax, betay, betaz);
  XYZTLorentzVector p(b * momentum());
  setMomentum(p.X(), p.Y(), p.Z(), p.T());
}

std::ostream& operator<<(std::ostream& o, const RawParticle& p) {
  o.setf(std::ios::fixed, std::ios::floatfield);
  o.setf(std::ios::right, std::ios::adjustfield);

  o << std::setw(4) << std::setprecision(2) << p.pid() << " (";
  o << std::setw(2) << std::setprecision(2) << p.status() << "): ";
  o << std::setw(10) << std::setprecision(4) << p.momentum() << " ";
  o << std::setw(10) << std::setprecision(4) << p.vertex();
  return o;
}

double RawParticle::et() const {
  double mypp, tmpEt = -1.;

  mypp = std::sqrt(momentum().mag2());
  if (mypp != 0) {
    tmpEt = E() * pt() / mypp;
  }
  return tmpEt;
}
