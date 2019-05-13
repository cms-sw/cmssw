
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayDerivatives.h"
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayModel.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>

TwoBodyDecayDerivatives::TwoBodyDecayDerivatives(double mPrimary, double mSecondary)
    : thePrimaryMass(mPrimary), theSecondaryMass(mSecondary) {}

TwoBodyDecayDerivatives::~TwoBodyDecayDerivatives() {}

const std::pair<AlgebraicMatrix, AlgebraicMatrix> TwoBodyDecayDerivatives::derivatives(const TwoBodyDecay &tbd) const {
  return derivatives(tbd.decayParameters());
}

const std::pair<AlgebraicMatrix, AlgebraicMatrix> TwoBodyDecayDerivatives::derivatives(
    const TwoBodyDecayParameters &param) const {
  // get the derivatives with respect to all parameters
  std::pair<AlgebraicMatrix, AlgebraicMatrix> dqsdpx = this->dqsdpx(param);
  std::pair<AlgebraicMatrix, AlgebraicMatrix> dqsdpy = this->dqsdpy(param);
  std::pair<AlgebraicMatrix, AlgebraicMatrix> dqsdpz = this->dqsdpz(param);
  std::pair<AlgebraicMatrix, AlgebraicMatrix> dqsdtheta = this->dqsdtheta(param);
  std::pair<AlgebraicMatrix, AlgebraicMatrix> dqsdphi = this->dqsdphi(param);
  std::pair<AlgebraicMatrix, AlgebraicMatrix> dqsdm = this->dqsdm(param);

  AlgebraicMatrix dqplusdz(3, dimension);
  dqplusdz.sub(1, px, dqsdpx.first);
  dqplusdz.sub(1, py, dqsdpy.first);
  dqplusdz.sub(1, pz, dqsdpz.first);
  dqplusdz.sub(1, theta, dqsdtheta.first);
  dqplusdz.sub(1, phi, dqsdphi.first);
  dqplusdz.sub(1, mass, dqsdm.first);

  AlgebraicMatrix dqminusdz(3, dimension);
  dqminusdz.sub(1, px, dqsdpx.second);
  dqminusdz.sub(1, py, dqsdpy.second);
  dqminusdz.sub(1, pz, dqsdpz.second);
  dqminusdz.sub(1, theta, dqsdtheta.second);
  dqminusdz.sub(1, phi, dqsdphi.second);
  dqminusdz.sub(1, mass, dqsdm.second);

  return std::make_pair(dqplusdz, dqminusdz);
}

const std::pair<AlgebraicMatrix, AlgebraicMatrix> TwoBodyDecayDerivatives::selectedDerivatives(
    const TwoBodyDecay &tbd, const std::vector<bool> &selector) const {
  return selectedDerivatives(tbd.decayParameters(), selector);
}

const std::pair<AlgebraicMatrix, AlgebraicMatrix> TwoBodyDecayDerivatives::selectedDerivatives(
    const TwoBodyDecayParameters &param, const std::vector<bool> &selector) const {
  if (selector.size() != dimension) {
    throw cms::Exception("BadConfig") << "@SUB=TwoBodyDecayDerivatives::selectedDerivatives"
                                      << "selector has bad dimension (size=" << selector.size() << ").";
  }

  int nSelected = std::count(selector.begin(), selector.end(), true);
  int iSelected = 1;

  AlgebraicMatrix dqplusdz(3, nSelected);
  AlgebraicMatrix dqminusdz(3, nSelected);
  std::pair<AlgebraicMatrix, AlgebraicMatrix> dqsdzi;

  for (unsigned int i = 1; i <= dimension; i++) {
    if (selector[i]) {
      dqsdzi = this->dqsdzi(param, DerivativeParameterName(i));
      dqplusdz.sub(1, iSelected, dqsdzi.first);
      dqminusdz.sub(1, iSelected, dqsdzi.second);
      iSelected++;
    }
  }

  return std::make_pair(dqplusdz, dqminusdz);
}

const std::pair<AlgebraicMatrix, AlgebraicMatrix> TwoBodyDecayDerivatives::dqsdpx(
    const TwoBodyDecayParameters &param) const {
  double px = param[TwoBodyDecayParameters::px];
  double py = param[TwoBodyDecayParameters::py];
  double pz = param[TwoBodyDecayParameters::pz];
  double theta = param[TwoBodyDecayParameters::theta];
  double phi = param[TwoBodyDecayParameters::phi];

  // compute transverse and absolute momentum
  double pT2 = px * px + py * py;
  double p2 = pT2 + pz * pz;
  double pT = sqrt(pT2);
  double p = sqrt(p2);

  double sphi = sin(phi);
  double cphi = cos(phi);
  double stheta = sin(theta);
  double ctheta = cos(theta);

  // some constants from kinematics
  double c1 = 0.5 * thePrimaryMass / theSecondaryMass;
  double c2 = sqrt(c1 * c1 - 1.);
  double c3 = 0.5 * c2 * ctheta / c1;
  double c4 = sqrt(p2 + thePrimaryMass * thePrimaryMass);

  // momentum of decay particle 1 in the primary's boosted frame
  AlgebraicMatrix pplus(3, 1);
  pplus[0][0] = theSecondaryMass * c2 * stheta * cphi;
  pplus[1][0] = theSecondaryMass * c2 * stheta * sphi;
  pplus[2][0] = 0.5 * p + c3 * c4;

  // momentum of decay particle 2 in the primary's boosted frame
  AlgebraicMatrix pminus(3, 1);
  pminus[0][0] = -pplus[0][0];
  pminus[1][0] = -pplus[1][0];
  pminus[2][0] = 0.5 * p - c3 * c4;

  // derivative of rotation matrix w.r.t. px
  AlgebraicMatrix dRotMatdpx(3, 3);

  dRotMatdpx[0][0] = pz / (pT * p) * (1. - px * px * (1. / pT2 + 1. / p2));
  dRotMatdpx[0][1] = px * py / (pT * pT2);
  dRotMatdpx[0][2] = (1. - px * px / p2) / p;

  dRotMatdpx[1][0] = -px * py * pz / (pT * p) * (1. / pT2 + 1. / p2);
  dRotMatdpx[1][1] = (1. - px * px / pT2) / pT;
  dRotMatdpx[1][2] = -px * py / (p * p2);

  dRotMatdpx[2][0] = -(1. / pT - pT / p2) * px / p;
  dRotMatdpx[2][1] = 0.;
  dRotMatdpx[2][2] = -px * pz / (p * p2);

  // derivative of the momentum of particle 1 in the lab frame w.r.t. px
  double dpplusdpx = px * (0.5 / p + c3 / c4);

  AlgebraicMatrix dqplusdpx = dRotMatdpx * pplus;
  dqplusdpx[0][0] += px * dpplusdpx / p;
  dqplusdpx[1][0] += py * dpplusdpx / p;
  dqplusdpx[2][0] += pz * dpplusdpx / p;

  // derivative of the momentum of particle 2 in the lab frame w.r.t. px
  double dpminusdpx = px * (0.5 / p - c3 / c4);

  AlgebraicMatrix dqminusdpx = dRotMatdpx * pminus;
  dqminusdpx[0][0] += px * dpminusdpx / p;
  dqminusdpx[1][0] += py * dpminusdpx / p;
  dqminusdpx[2][0] += pz * dpminusdpx / p;

  // return result
  return std::make_pair(dqplusdpx, dqminusdpx);
}

const std::pair<AlgebraicMatrix, AlgebraicMatrix> TwoBodyDecayDerivatives::dqsdpy(
    const TwoBodyDecayParameters &param) const {
  double px = param[TwoBodyDecayParameters::px];
  double py = param[TwoBodyDecayParameters::py];
  double pz = param[TwoBodyDecayParameters::pz];
  double theta = param[TwoBodyDecayParameters::theta];
  double phi = param[TwoBodyDecayParameters::phi];

  // compute transverse and absolute momentum
  double pT2 = px * px + py * py;
  double p2 = pT2 + pz * pz;
  double pT = sqrt(pT2);
  double p = sqrt(p2);

  double sphi = sin(phi);
  double cphi = cos(phi);
  double stheta = sin(theta);
  double ctheta = cos(theta);

  // some constants from kinematics
  double c1 = 0.5 * thePrimaryMass / theSecondaryMass;
  double c2 = sqrt(c1 * c1 - 1.);
  double c3 = 0.5 * c2 * ctheta / c1;
  double c4 = sqrt(p2 + thePrimaryMass * thePrimaryMass);

  // momentum of decay particle 1 in the rest frame of the primary
  AlgebraicMatrix pplus(3, 1);
  pplus[0][0] = theSecondaryMass * c2 * stheta * cphi;
  pplus[1][0] = theSecondaryMass * c2 * stheta * sphi;
  pplus[2][0] = 0.5 * p + c3 * c4;

  // momentum of decay particle 2 in the rest frame of the primary
  AlgebraicMatrix pminus(3, 1);
  pminus[0][0] = -pplus[0][0];
  pminus[1][0] = -pplus[1][0];
  pminus[2][0] = 0.5 * p - c3 * c4;

  // derivative of rotation matrix w.r.t. py
  AlgebraicMatrix dRotMatdpy(3, 3);

  dRotMatdpy[0][0] = -px * py * pz / (pT * p) * (1. / pT2 + 1. / p2);
  dRotMatdpy[0][1] = (py * py / pT2 - 1.) / pT;
  dRotMatdpy[0][2] = -px * py / (p * p2);

  dRotMatdpy[1][0] = pz / (pT * p) * (1. - py * py * (1. / pT2 + 1. / p2));
  dRotMatdpy[1][1] = -px * py / (pT * pT2);
  dRotMatdpy[1][2] = (1. - py * py / p2) / p;

  dRotMatdpy[2][0] = -(1. / pT - pT / p2) * py / p;
  dRotMatdpy[2][1] = 0.;
  dRotMatdpy[2][2] = -py * pz / (p * p2);

  // derivative of the momentum of particle 1 in the lab frame w.r.t. py
  double dpplusdpy = py * (0.5 / p + c3 / c4);

  AlgebraicMatrix dqplusdpy = dRotMatdpy * pplus;
  dqplusdpy[0][0] += px * dpplusdpy / p;
  dqplusdpy[1][0] += py * dpplusdpy / p;
  dqplusdpy[2][0] += pz * dpplusdpy / p;

  // derivative of the momentum of particle 2 in the lab frame w.r.t. py
  double dpminusdpy = py * (0.5 / p - c3 / c4);

  AlgebraicMatrix dqminusdpy = dRotMatdpy * pminus;
  dqminusdpy[0][0] += px * dpminusdpy / p;
  dqminusdpy[1][0] += py * dpminusdpy / p;
  dqminusdpy[2][0] += pz * dpminusdpy / p;

  // return result
  return std::make_pair(dqplusdpy, dqminusdpy);
}

const std::pair<AlgebraicMatrix, AlgebraicMatrix> TwoBodyDecayDerivatives::dqsdpz(
    const TwoBodyDecayParameters &param) const {
  double px = param[TwoBodyDecayParameters::px];
  double py = param[TwoBodyDecayParameters::py];
  double pz = param[TwoBodyDecayParameters::pz];
  double theta = param[TwoBodyDecayParameters::theta];
  double phi = param[TwoBodyDecayParameters::phi];

  // compute transverse and absolute momentum
  double pT2 = px * px + py * py;
  double p2 = pT2 + pz * pz;
  double pT = sqrt(pT2);
  double p = sqrt(p2);

  double sphi = sin(phi);
  double cphi = cos(phi);
  double stheta = sin(theta);
  double ctheta = cos(theta);

  // some constants from kinematics
  double c1 = 0.5 * thePrimaryMass / theSecondaryMass;
  double c2 = sqrt(c1 * c1 - 1.);
  double c3 = 0.5 * c2 * ctheta / c1;
  double c4 = sqrt(p2 + thePrimaryMass * thePrimaryMass);

  // momentum of decay particle 1 in the rest frame of the primary
  AlgebraicMatrix pplus(3, 1);
  pplus[0][0] = theSecondaryMass * c2 * stheta * cphi;
  pplus[1][0] = theSecondaryMass * c2 * stheta * sphi;
  pplus[2][0] = 0.5 * p + c3 * c4;

  // momentum of decay particle 2 in the rest frame of the primary
  AlgebraicMatrix pminus(3, 1);
  pminus[0][0] = -pplus[0][0];
  pminus[1][0] = -pplus[1][0];
  pminus[2][0] = 0.5 * p - c3 * c4;

  // derivative of rotation matrix w.r.t. py
  AlgebraicMatrix dRotMatdpz(3, 3);

  dRotMatdpz[0][0] = px / (pT * p) * (1. - pz * pz / p2);
  dRotMatdpz[0][1] = 0.;
  dRotMatdpz[0][2] = -px * pz / (p * p2);

  dRotMatdpz[1][0] = py / (p * pT) * (1. - pz * pz / p2);
  dRotMatdpz[1][1] = 0.;
  dRotMatdpz[1][2] = -py * pz / (p * p2);

  dRotMatdpz[2][0] = pT * pz / (p * p2);
  dRotMatdpz[2][1] = 0.;
  dRotMatdpz[2][2] = (1. - pz * pz / p2) / p;

  // derivative of the momentum of particle 1 in the lab frame w.r.t. pz
  double dpplusdpz = pz * (0.5 / p + c3 / c4);

  AlgebraicMatrix dqplusdpz = dRotMatdpz * pplus;
  dqplusdpz[0][0] += px * dpplusdpz / p;
  dqplusdpz[1][0] += py * dpplusdpz / p;
  dqplusdpz[2][0] += pz * dpplusdpz / p;

  // derivative of the momentum of particle 2 in the lab frame w.r.t. pz
  double dpminusdpz = pz * (0.5 / p - c3 / c4);

  AlgebraicMatrix dqminusdpz = dRotMatdpz * pminus;
  dqminusdpz[0][0] += px * dpminusdpz / p;
  dqminusdpz[1][0] += py * dpminusdpz / p;
  dqminusdpz[2][0] += pz * dpminusdpz / p;

  // return result
  return std::make_pair(dqplusdpz, dqminusdpz);
}

const std::pair<AlgebraicMatrix, AlgebraicMatrix> TwoBodyDecayDerivatives::dqsdtheta(
    const TwoBodyDecayParameters &param) const {
  double px = param[TwoBodyDecayParameters::px];
  double py = param[TwoBodyDecayParameters::py];
  double pz = param[TwoBodyDecayParameters::pz];
  double theta = param[TwoBodyDecayParameters::theta];
  double phi = param[TwoBodyDecayParameters::phi];

  // compute transverse and absolute momentum
  double pT2 = px * px + py * py;
  double p2 = pT2 + pz * pz;

  double sphi = sin(phi);
  double cphi = cos(phi);
  double stheta = sin(theta);
  double ctheta = cos(theta);

  // some constants from kinematics
  double c1 = 0.5 * thePrimaryMass / theSecondaryMass;
  double c2 = sqrt(c1 * c1 - 1.);
  double c3 = -0.5 * c2 * stheta / c1;
  double c4 = sqrt(p2 + thePrimaryMass * thePrimaryMass);

  // derivative of the momentum of particle 1 in the primary's rest frame w.r.t.
  // theta
  AlgebraicMatrix dpplusdtheta(3, 1);
  dpplusdtheta[0][0] = theSecondaryMass * c2 * ctheta * cphi;
  dpplusdtheta[1][0] = theSecondaryMass * c2 * ctheta * sphi;
  dpplusdtheta[2][0] = c3 * c4;

  // derivative of the momentum of particle 2 in the primary's rest frame w.r.t.
  // theta
  AlgebraicMatrix dpminusdtheta(3, 1);
  dpminusdtheta[0][0] = -theSecondaryMass * c2 * ctheta * cphi;
  dpminusdtheta[1][0] = -theSecondaryMass * c2 * ctheta * sphi;
  dpminusdtheta[2][0] = -c3 * c4;

  TwoBodyDecayModel decayModel;
  AlgebraicMatrix rotMat = decayModel.rotationMatrix(px, py, pz);

  AlgebraicMatrix dqplusdtheta = rotMat * dpplusdtheta;
  AlgebraicMatrix dqminusdtheta = rotMat * dpminusdtheta;

  return std::make_pair(dqplusdtheta, dqminusdtheta);
}

const std::pair<AlgebraicMatrix, AlgebraicMatrix> TwoBodyDecayDerivatives::dqsdphi(
    const TwoBodyDecayParameters &param) const {
  double px = param[TwoBodyDecayParameters::px];
  double py = param[TwoBodyDecayParameters::py];
  double pz = param[TwoBodyDecayParameters::pz];
  double theta = param[TwoBodyDecayParameters::theta];
  double phi = param[TwoBodyDecayParameters::phi];

  double sphi = sin(phi);
  double cphi = cos(phi);
  double stheta = sin(theta);

  // some constants from kinematics
  double c1 = 0.5 * thePrimaryMass / theSecondaryMass;
  double c2 = sqrt(c1 * c1 - 1.);

  // derivative of the momentum of particle 1 in the primary's rest frame w.r.t.
  // phi
  AlgebraicMatrix dpplusdphi(3, 1);
  dpplusdphi[0][0] = -theSecondaryMass * c2 * stheta * sphi;
  dpplusdphi[1][0] = theSecondaryMass * c2 * stheta * cphi;
  dpplusdphi[2][0] = 0.;

  // derivative of the momentum of particle 2 in the primary's rest frame w.r.t.
  // phi
  AlgebraicMatrix dpminusdphi(3, 1);
  dpminusdphi[0][0] = theSecondaryMass * c2 * stheta * sphi;
  dpminusdphi[1][0] = -theSecondaryMass * c2 * stheta * cphi;
  dpminusdphi[2][0] = 0.;

  TwoBodyDecayModel decayModel;
  AlgebraicMatrix rotMat = decayModel.rotationMatrix(px, py, pz);

  AlgebraicMatrix dqplusdphi = rotMat * dpplusdphi;
  AlgebraicMatrix dqminusdphi = rotMat * dpminusdphi;

  return std::make_pair(dqplusdphi, dqminusdphi);
}

const std::pair<AlgebraicMatrix, AlgebraicMatrix> TwoBodyDecayDerivatives::dqsdm(
    const TwoBodyDecayParameters &param) const {
  double px = param[TwoBodyDecayParameters::px];
  double py = param[TwoBodyDecayParameters::py];
  double pz = param[TwoBodyDecayParameters::pz];
  double theta = param[TwoBodyDecayParameters::theta];
  double phi = param[TwoBodyDecayParameters::phi];

  double pT2 = px * px + py * py;
  double p2 = pT2 + pz * pz;

  double sphi = sin(phi);
  double cphi = cos(phi);
  double ctheta = cos(theta);
  double stheta = sin(theta);

  // some constants from kinematics
  double c1 = 0.5 * thePrimaryMass / theSecondaryMass;
  double c2 = 1. / sqrt(c1 * c1 - 1.);
  double m2 = thePrimaryMass * thePrimaryMass;

  // derivative of the momentum of particle 1 in the primary's rest frame w.r.t.
  // the primary's mass
  AlgebraicMatrix dpplusdm(3, 1);
  dpplusdm[0][0] = c2 * 0.5 * c1 * stheta * cphi;
  dpplusdm[1][0] = c2 * 0.5 * c1 * stheta * sphi;
  dpplusdm[2][0] = c2 * theSecondaryMass * (c1 * c1 + p2 / m2) / sqrt(p2 + m2) * ctheta;

  // derivative of the momentum of particle 2 in the primary's rest frame w.r.t.
  // the primary's mass
  AlgebraicMatrix dpminusdm(3, 1);
  dpminusdm[0][0] = -dpplusdm[0][0];
  dpminusdm[1][0] = -dpplusdm[1][0];
  dpminusdm[2][0] = -dpplusdm[2][0];

  TwoBodyDecayModel decayModel;
  AlgebraicMatrix rotMat = decayModel.rotationMatrix(px, py, pz);

  AlgebraicMatrix dqplusdm = rotMat * dpplusdm;
  AlgebraicMatrix dqminusdm = rotMat * dpminusdm;

  return std::make_pair(dqplusdm, dqminusdm);
}

const std::pair<AlgebraicMatrix, AlgebraicMatrix> TwoBodyDecayDerivatives::dqsdzi(
    const TwoBodyDecayParameters &param, const DerivativeParameterName &i) const {
  switch (i) {
    case TwoBodyDecayDerivatives::px:
      return dqsdpx(param);
      break;
    case TwoBodyDecayDerivatives::py:
      return dqsdpy(param);
      break;
    case TwoBodyDecayDerivatives::pz:
      return dqsdpz(param);
      break;
    case TwoBodyDecayDerivatives::theta:
      return dqsdtheta(param);
      break;
    case TwoBodyDecayDerivatives::phi:
      return dqsdphi(param);
      break;
    case TwoBodyDecayDerivatives::mass:
      return dqsdm(param);
      break;
    default:
      throw cms::Exception("BadConfig") << "@SUB=TwoBodyDecayDerivatives::dqsdzi"
                                        << "no decay parameter related to selector (" << i << ").";
  };

  return std::make_pair(AlgebraicMatrix(3, 1, 0), AlgebraicMatrix(3, 1, 0));
}
