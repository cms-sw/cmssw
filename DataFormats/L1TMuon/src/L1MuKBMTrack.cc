#include "DataFormats/L1TMuon/interface/L1MuKBMTrack.h"

L1MuKBMTrack::L1MuKBMTrack() : reco::LeafCandidate(-1, reco::LeafCandidate::PolarLorentzVector(0.1, 0.0, 0.0, 0.105)) {}

L1MuKBMTrack::~L1MuKBMTrack() {}

L1MuKBMTrack::L1MuKBMTrack(const L1MuKBMTCombinedStubRef& seed, int phi, int phiB)
    : reco::LeafCandidate(-1, reco::LeafCandidate::PolarLorentzVector(0.1, 0.0, 0.0, 0.105)),
      covariance_(std::vector<double>(6, 0.0)),
      curvVertex_(0),
      phiVertex_(0),
      dxy_(0),
      curvMuon_(0),
      phiMuon_(0),
      phiBMuon_(0),
      curv_(0),
      phi_(phi),
      phiB_(phiB),
      coarseEta_(0),
      approxChi2_(0),
      trackCompatibility_(0),
      hitPattern_(0),
      step_(seed->stNum()),
      sector_(seed->scNum()),
      wheel_(seed->whNum()),
      quality_(seed->quality()),
      hasFineEta_(false),
      bx_(seed->bxNum()),
      rank_(seed->bxNum()),
      ptUnconstrained_(0.0) {
  stubs_.push_back(seed);
  residuals_.push_back(0);
  residuals_.push_back(0);
  residuals_.push_back(0);
}

int L1MuKBMTrack::curvatureAtMuon() const { return curvMuon_; }
int L1MuKBMTrack::phiAtMuon() const { return phiMuon_; }
int L1MuKBMTrack::phiBAtMuon() const { return phiBMuon_; }

int L1MuKBMTrack::curvatureAtVertex() const { return curvVertex_; }

int L1MuKBMTrack::phiAtVertex() const { return phiVertex_; }

int L1MuKBMTrack::dxy() const { return dxy_; }

int L1MuKBMTrack::curvature() const { return curv_; }

int L1MuKBMTrack::positionAngle() const { return phi_; }

int L1MuKBMTrack::bendingAngle() const { return phiB_; }

int L1MuKBMTrack::coarseEta() const { return coarseEta_; }

int L1MuKBMTrack::approxChi2() const { return approxChi2_; }
int L1MuKBMTrack::trackCompatibility() const { return trackCompatibility_; }

int L1MuKBMTrack::hitPattern() const { return hitPattern_; }

int L1MuKBMTrack::step() const { return step_; }
int L1MuKBMTrack::sector() const { return sector_; }
int L1MuKBMTrack::wheel() const { return wheel_; }

int L1MuKBMTrack::quality() const { return quality_; }

float L1MuKBMTrack::ptUnconstrained() const { return ptUnconstrained_; }

int L1MuKBMTrack::fineEta() const { return fineEta_; }

bool L1MuKBMTrack::hasFineEta() const { return hasFineEta_; }

int L1MuKBMTrack::bx() const { return bx_; }

int L1MuKBMTrack::rank() const { return rank_; }

const L1MuKBMTCombinedStubRefVector& L1MuKBMTrack::stubs() const { return stubs_; }

int L1MuKBMTrack::residual(uint i) const { return residuals_[i]; }

void L1MuKBMTrack::setCoordinates(int step, int curv, int phi, int phiB) {
  step_ = step;
  curv_ = curv;
  phiB_ = phiB;
  phi_ = phi;
}

void L1MuKBMTrack::setCoordinatesAtVertex(int curv, int phi, int dxy) {
  curvVertex_ = curv;
  phiVertex_ = phi;
  dxy_ = dxy;
}

void L1MuKBMTrack::setCoordinatesAtMuon(int curv, int phi, int phiB) {
  curvMuon_ = curv;
  phiMuon_ = phi;
  phiBMuon_ = phiB;
}

void L1MuKBMTrack::setCoarseEta(int eta) { coarseEta_ = eta; }

void L1MuKBMTrack::setHitPattern(int pattern) { hitPattern_ = pattern; }

void L1MuKBMTrack::setApproxChi2(int chi) { approxChi2_ = chi; }
void L1MuKBMTrack::setTrackCompatibility(int chi) { trackCompatibility_ = chi; }

void L1MuKBMTrack::setPtEtaPhi(double pt, double eta, double phi) {
  PolarLorentzVector v(pt, eta, phi, 0.105);
  setP4(v);
}
void L1MuKBMTrack::setPtUnconstrained(float pt) { ptUnconstrained_ = pt; }

void L1MuKBMTrack::addStub(const L1MuKBMTCombinedStubRef& stub) {
  if (stub->quality() < quality_)
    quality_ = stub->quality();
  stubs_.push_back(stub);
}

void L1MuKBMTrack::setFineEta(int eta) {
  fineEta_ = eta;
  hasFineEta_ = true;
}

void L1MuKBMTrack::setRank(int rank) { rank_ = rank; }

void L1MuKBMTrack::setKalmanGain(
    unsigned int step, unsigned int K, float a1, float a2, float a3, float a4, float a5, float a6) {
  switch (step) {
    case 3:
      kalmanGain3_.push_back(K);
      kalmanGain3_.push_back(a1);
      kalmanGain3_.push_back(a2);
      kalmanGain3_.push_back(a3);
      kalmanGain3_.push_back(a4);
      kalmanGain3_.push_back(a5);
      kalmanGain3_.push_back(a6);
      break;
    case 2:
      kalmanGain2_.push_back(K);
      kalmanGain2_.push_back(a1);
      kalmanGain2_.push_back(a2);
      kalmanGain2_.push_back(a3);
      kalmanGain2_.push_back(a4);
      kalmanGain2_.push_back(a5);
      kalmanGain2_.push_back(a6);
      break;
    case 1:
      kalmanGain1_.push_back(K);
      kalmanGain1_.push_back(a1);
      kalmanGain1_.push_back(a2);
      kalmanGain1_.push_back(a3);
      kalmanGain1_.push_back(a4);
      kalmanGain1_.push_back(a5);
      kalmanGain1_.push_back(a6);
      break;
    case 0:
      kalmanGain0_.push_back(K);
      kalmanGain0_.push_back(a1);
      kalmanGain0_.push_back(a2);
      kalmanGain0_.push_back(a3);
      break;

    default:
      printf("Critical ERROR on setting the Klamn gain\n");
  }
}

void L1MuKBMTrack::setResidual(uint i, int val) { residuals_[i] = val; }

const std::vector<float>& L1MuKBMTrack::kalmanGain(unsigned int step) const {
  switch (step) {
    case 3:
      return kalmanGain3_;
    case 2:
      return kalmanGain2_;
    case 1:
      return kalmanGain1_;
    case 0:
      return kalmanGain0_;
  }
  return kalmanGain0_;
}

const std::vector<double>& L1MuKBMTrack::covariance() const { return covariance_; }

bool L1MuKBMTrack::overlapTrack(const L1MuKBMTrack& other) const {
  for (const auto& s1 : stubs_) {
    for (const auto& s2 : other.stubs()) {
      if (s1->scNum() == s2->scNum() && s1->whNum() == s2->whNum() && s1->stNum() == s2->stNum() &&
          s1->tag() == s2->tag())
        return true;
    }
  }
  return false;
}

void L1MuKBMTrack::setCovariance(const CovarianceMatrix& c) {
  covariance_[0] = c(0, 0);
  covariance_[1] = c(0, 1);
  covariance_[2] = c(1, 1);
  covariance_[3] = c(0, 2);
  covariance_[4] = c(1, 2);
  covariance_[5] = c(2, 2);
}
