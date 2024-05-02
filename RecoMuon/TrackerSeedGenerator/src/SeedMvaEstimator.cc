#include "RecoMuon/TrackerSeedGenerator/interface/SeedMvaEstimator.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CommonTools/MVAUtils/interface/GBRForestTools.h"

SeedMvaEstimator::SeedMvaEstimator(const edm::FileInPath& weightsfile,
                                   const std::vector<double>& scale_mean,
                                   const std::vector<double>& scale_std,
                                   const bool isFromL1,
                                   const int minL1Qual)
    : scale_mean_(scale_mean.begin(), scale_mean.end()), isFromL1_(isFromL1), minL1Qual_(minL1Qual) {
  scale_istd_.reserve(scale_std.size());
  for (auto v : scale_std)
    scale_istd_.push_back(1. / v);
  gbrForest_ = createGBRForest(weightsfile);
}

SeedMvaEstimator::~SeedMvaEstimator() {}

namespace {
  enum inputIndexes {
    kTsosErr0,       // 0
    kTsosErr2,       // 1
    kTsosErr5,       // 2
    kTsosDxdz,       // 3
    kTsosDydz,       // 4
    kTsosQbp,        // 5
    kDRdRL1SeedP,    // 6
    kDPhidRL1SeedP,  // 7
    kLastL1,         // 8

    kDRdRL2SeedP = 8,  // 8
    kDPhidRL2SeedP,    // 9
    kLastL2,           // 10
  };
}  // namespace

void SeedMvaEstimator::getL1MuonVariables(const GlobalVector& global_p,
                                          const l1t::MuonBxCollection& l1Muons,
                                          float& dR2dRL1SeedP,
                                          float& dPhidRL1SeedP) const {
  int ibx = 0;  // -- only take when ibx == 0 -- //

  auto eta = global_p.eta();
  auto phi = global_p.barePhi();
  for (auto it = l1Muons.begin(ibx); it != l1Muons.end(ibx); it++) {
    if (it->hwQual() < minL1Qual_)
      continue;

    float dR2tmp = reco::deltaR2(it->etaAtVtx(), it->phiAtVtx(), eta, phi);
    if (dR2tmp < dR2dRL1SeedP) {
      dR2dRL1SeedP = dR2tmp;
      dPhidRL1SeedP = reco::deltaPhi(it->phiAtVtx(), phi);
    }
  }
}

void SeedMvaEstimator::getL2MuonVariables(const GlobalVector& global_p,
                                          const reco::RecoChargedCandidateCollection& l2Muons,
                                          float& dR2dRL2SeedP,
                                          float& dPhidRL2SeedP) const {
  auto eta = global_p.eta();
  auto phi = global_p.barePhi();
  for (auto it = l2Muons.begin(); it != l2Muons.end(); it++) {
    float dR2tmp = reco::deltaR2(it->eta(), it->phi(), eta, phi);
    if (dR2tmp < dR2dRL2SeedP) {
      dR2dRL2SeedP = dR2tmp;
      dPhidRL2SeedP = reco::deltaPhi(it->phi(), phi);
    }
  }
}

double SeedMvaEstimator::computeMva(const TrajectorySeed& seed,
                                    const GlobalVector& global_p,
                                    const l1t::MuonBxCollection& l1Muons,
                                    const reco::RecoChargedCandidateCollection& l2Muons) const {
  static constexpr float initDRdPhi(99999.);
  auto kLast = isFromL1_ ? kLastL1 : kLastL2;
  float var[kLast];

  var[kTsosErr0] = seed.startingState().error(0);
  var[kTsosErr2] = seed.startingState().error(2);
  var[kTsosErr5] = seed.startingState().error(5);
  var[kTsosDxdz] = seed.startingState().parameters().dxdz();
  var[kTsosDydz] = seed.startingState().parameters().dydz();
  var[kTsosQbp] = seed.startingState().parameters().qbp();

  float dR2dRL1SeedP = initDRdPhi;
  float dPhidRL1SeedP = initDRdPhi;
  getL1MuonVariables(global_p, l1Muons, dR2dRL1SeedP, dPhidRL1SeedP);

  var[kDRdRL1SeedP] = std::sqrt(dR2dRL1SeedP);
  var[kDPhidRL1SeedP] = dPhidRL1SeedP;

  if (!isFromL1_) {
    float dR2dRL2SeedP = initDRdPhi;
    float dPhidRL2SeedP = initDRdPhi;
    getL2MuonVariables(global_p, l2Muons, dR2dRL2SeedP, dPhidRL2SeedP);

    var[kDRdRL2SeedP] = std::sqrt(dR2dRL2SeedP);
    var[kDPhidRL2SeedP] = dPhidRL2SeedP;
  }

  for (int iv = 0; iv < kLast; ++iv) {
    var[iv] = (var[iv] - scale_mean_[iv]) * scale_istd_[iv];
  }

  return gbrForest_->GetResponse(var);
}
