#include "DataFormats/L1THGCal/interface/ClusterShapes.h"
#include <cmath>

using namespace l1t;

ClusterShapes ClusterShapes::operator+(const ClusterShapes& x) {
  ClusterShapes cs(*this);  // copy constructor
  cs += x;
  return cs;
}

void ClusterShapes::operator+=(const ClusterShapes& x) {
  sum_e_ += x.sum_e_;
  sum_e2_ += x.sum_e2_;
  sum_logE_ += x.sum_logE_;
  n_ += x.n_;

  sum_w_ += x.sum_w_;

  emax_ = (emax_ > x.emax_) ? emax_ : x.emax_;

  // mid-point
  sum_eta_ += x.sum_eta_;
  sum_phi_0_ += x.sum_phi_0_;  //
  sum_phi_1_ += x.sum_phi_1_;  //
  sum_r_ += x.sum_r_;

  // square
  sum_eta2_ += x.sum_eta2_;
  sum_phi2_0_ += x.sum_phi2_0_;
  sum_phi2_1_ += x.sum_phi2_1_;
  sum_r2_ += x.sum_r2_;

  // off diagonal
  sum_eta_r_ += x.sum_eta_r_;
  sum_r_phi_0_ += x.sum_r_phi_0_;
  sum_r_phi_1_ += x.sum_r_phi_1_;
  sum_eta_phi_0_ += x.sum_eta_phi_0_;
  sum_eta_phi_1_ += x.sum_eta_phi_1_;
}

// -------------- CLUSTER SHAPES ---------------
void ClusterShapes::Init(float e, float eta, float phi, float r) {
  if (e <= 0)
    return;
  sum_e_ = e;
  sum_e2_ = e * e;
  sum_logE_ = std::log(e);

  float w = e;

  n_ = 1;

  sum_w_ = w;

  sum_phi_0_ = w * (phi);
  sum_phi_1_ = w * (phi + M_PI);
  sum_r_ = w * r;
  sum_eta_ = w * eta;

  //--
  sum_r2_ += w * (r * r);
  sum_eta2_ += w * (eta * eta);
  sum_phi2_0_ += w * (phi * phi);
  sum_phi2_1_ += w * (phi + M_PI) * (phi + M_PI);

  // -- off diagonal
  sum_eta_r_ += w * (r * eta);
  sum_r_phi_0_ += w * (r * phi);
  sum_r_phi_1_ += w * r * (phi + M_PI);
  sum_eta_phi_0_ += w * (eta * phi);
  sum_eta_phi_1_ += w * eta * (phi + M_PI);
}
// ------
float ClusterShapes::Eta() const { return sum_eta_ / sum_w_; }
float ClusterShapes::R() const { return sum_r_ / sum_w_; }

float ClusterShapes::SigmaEtaEta() const { return sum_eta2_ / sum_w_ - Eta() * Eta(); }

float ClusterShapes::SigmaRR() const { return sum_r2_ / sum_w_ - R() * R(); }

float ClusterShapes::SigmaPhiPhi() const {
  float phi_0 = (sum_phi_0_ / sum_w_);
  float phi_1 = (sum_phi_1_ / sum_w_);
  float spp_0 = sum_phi2_0_ / sum_w_ - phi_0 * phi_0;
  float spp_1 = sum_phi2_1_ / sum_w_ - phi_1 * phi_1;

  if (spp_0 < spp_1) {
    isPhi0_ = true;
    return spp_0;
  } else {
    isPhi0_ = false;
    return spp_1;
  }
}

float ClusterShapes::Phi() const {
  SigmaPhiPhi();  //update phi
  if (isPhi0_)
    return (sum_phi_0_ / sum_w_);
  else
    return (sum_phi_1_ / sum_w_);
}

// off - diagonal
float ClusterShapes::SigmaEtaR() const { return -(sum_eta_r_ / sum_w_ - Eta() * R()); }

float ClusterShapes::SigmaEtaPhi() const {
  SigmaPhiPhi();  // decide which phi use, update phi

  if (isPhi0_)
    return -(sum_eta_phi_0_ / sum_w_ - Eta() * (sum_phi_0_ / sum_w_));
  else
    return -(sum_eta_phi_1_ / sum_w_ - Eta() * (sum_phi_1_ / sum_w_));
}

float ClusterShapes::SigmaRPhi() const {
  SigmaPhiPhi();  // decide which phi use, update phi
  if (isPhi0_)
    return -(sum_r_phi_0_ / sum_w_ - R() * (sum_phi_0_ / sum_w_));
  else
    return -(sum_r_phi_1_ / sum_w_ - R() * (sum_phi_1_ / sum_w_));
}
