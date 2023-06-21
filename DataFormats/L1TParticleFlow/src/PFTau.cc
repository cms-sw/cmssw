#include "DataFormats/L1TParticleFlow/interface/PFTau.h"

l1t::PFTau::PFTau(
    const PolarLorentzVector& p, float NNValues[80], float iso, float fulliso, int id, int hwpt, int hweta, int hwphi)
    : L1Candidate(p, hwpt, hweta, hwphi, /*hwQuality=*/int(0)), iso_(iso), fullIso_(fulliso), id_(id) {
  for (int i0 = 0; i0 < 80; i0++)
    NNValues_[i0] = NNValues[i0];  // copy the array of NN inputs
}
