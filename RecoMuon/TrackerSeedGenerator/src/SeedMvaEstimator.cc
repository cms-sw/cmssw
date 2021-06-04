#include "RecoMuon/TrackerSeedGenerator/interface/SeedMvaEstimator.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CommonTools/MVAUtils/interface/GBRForestTools.h"

SeedMvaEstimator::SeedMvaEstimator(const edm::FileInPath& weightsfile,
                                   std::vector<double> scale_mean,
                                   std::vector<double> scale_std) {
  gbrForest_ = createGBRForest(weightsfile);
  scale_mean_ = scale_mean;
  scale_std_ = scale_std;
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

void SeedMvaEstimator::getL1MuonVariables(const TrajectorySeed& seed,
                                          GlobalVector global_p,
                                          edm::Handle<l1t::MuonBxCollection> h_L1Muon,
                                          int minL1Qual,
                                          float& dRdRL1SeedP,
                                          float& dPhidRL1SeedP) const {
  for (int ibx = h_L1Muon->getFirstBX(); ibx <= h_L1Muon->getLastBX(); ++ibx) {
    if (ibx != 0)
      continue;  // -- only take when ibx == 0 -- //
    for (auto it = h_L1Muon->begin(ibx); it != h_L1Muon->end(ibx); it++) {
      l1t::MuonRef ref_L1Mu(h_L1Muon, distance(h_L1Muon->begin(h_L1Muon->getFirstBX()), it));

      if (ref_L1Mu->hwQual() < minL1Qual)
        continue;

      float dR2tmp = reco::deltaR2(ref_L1Mu->etaAtVtx(), ref_L1Mu->phiAtVtx(), global_p.eta(), global_p.phi());

      if (dR2tmp < dRdRL1SeedP * dRdRL1SeedP) {
        dRdRL1SeedP = std::sqrt(dR2tmp);
        dPhidRL1SeedP = reco::deltaPhi(ref_L1Mu->phiAtVtx(), global_p.phi());
      }
    }
  }
}

void SeedMvaEstimator::getL2MuonVariables(const TrajectorySeed& seed,
                                          GlobalVector global_p,
                                          edm::Handle<reco::RecoChargedCandidateCollection> h_L2Muon,
                                          float& dRdRL2SeedP,
                                          float& dPhidRL2SeedP) const {
  for (unsigned int i_L2 = 0; i_L2 < h_L2Muon->size(); i_L2++) {
    reco::RecoChargedCandidateRef ref_L2Mu(h_L2Muon, i_L2);

    float dR2tmp = reco::deltaR2(*ref_L2Mu, global_p);

    if (dR2tmp < dRdRL2SeedP * dRdRL2SeedP) {
      dRdRL2SeedP = std::sqrt(dR2tmp);
      dPhidRL2SeedP = reco::deltaPhi(ref_L2Mu->phi(), global_p.phi());
    }
  }
}

double SeedMvaEstimator::computeMva(const TrajectorySeed& seed,
                                    GlobalVector global_p,
                                    edm::Handle<l1t::MuonBxCollection> h_L1Muon,
                                    int minL1Qual,
                                    edm::Handle<reco::RecoChargedCandidateCollection> h_L2Muon,
                                    bool isFromL1) const {
  float initDRdPhi = 99999.;
  if (isFromL1) {
    float var[kLastL1]{};

    var[kTsosErr0] = seed.startingState().error(0);
    var[kTsosErr2] = seed.startingState().error(2);
    var[kTsosErr5] = seed.startingState().error(5);
    var[kTsosDxdz] = seed.startingState().parameters().dxdz();
    var[kTsosDydz] = seed.startingState().parameters().dydz();
    var[kTsosQbp] = seed.startingState().parameters().qbp();

    float dRdRL1SeedP = initDRdPhi;
    float dPhidRL1SeedP = initDRdPhi;
    getL1MuonVariables(seed, global_p, h_L1Muon, minL1Qual, dRdRL1SeedP, dPhidRL1SeedP);

    var[kDRdRL1SeedP] = dRdRL1SeedP;
    var[kDPhidRL1SeedP] = dPhidRL1SeedP;

    for (int iv = 0; iv < kLastL1; ++iv) {
      var[iv] = (var[iv] - scale_mean_.at(iv)) / scale_std_.at(iv);
    }

    return gbrForest_->GetResponse(var);
  } else {
    float var[kLastL2]{};

    var[kTsosErr0] = seed.startingState().error(0);
    var[kTsosErr2] = seed.startingState().error(2);
    var[kTsosErr5] = seed.startingState().error(5);
    var[kTsosDxdz] = seed.startingState().parameters().dxdz();
    var[kTsosDydz] = seed.startingState().parameters().dydz();
    var[kTsosQbp] = seed.startingState().parameters().qbp();

    float dRdRL1SeedP = initDRdPhi;
    float dPhidRL1SeedP = initDRdPhi;
    getL1MuonVariables(seed, global_p, h_L1Muon, minL1Qual, dRdRL1SeedP, dPhidRL1SeedP);

    float dRdRL2SeedP = initDRdPhi;
    float dPhidRL2SeedP = initDRdPhi;
    getL2MuonVariables(seed, global_p, h_L2Muon, dRdRL2SeedP, dPhidRL2SeedP);

    var[kDRdRL1SeedP] = dRdRL1SeedP;
    var[kDPhidRL1SeedP] = dPhidRL1SeedP;
    var[kDRdRL2SeedP] = dRdRL2SeedP;
    var[kDPhidRL2SeedP] = dPhidRL2SeedP;

    for (int iv = 0; iv < kLastL2; ++iv) {
      var[iv] = (var[iv] - scale_mean_.at(iv)) / scale_std_.at(iv);
    }

    return gbrForest_->GetResponse(var);
  }
}
