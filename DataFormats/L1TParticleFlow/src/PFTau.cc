#include "DataFormats/L1TParticleFlow/interface/PFTau.h"

l1t::PFTau::PFTau(const PolarLorentzVector& p, float iso, float fulliso, int id, int hwpt, int hweta, int hwphi)
    : L1Candidate(p, hwpt, hweta, hwphi, /*hwQuality=*/int(0)), iso_(iso), fullIso_(fulliso), id_(id) {}
