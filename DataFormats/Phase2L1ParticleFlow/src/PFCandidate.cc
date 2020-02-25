#include "DataFormats/Phase2L1ParticleFlow/interface/PFCandidate.h"

l1t::PFCandidate::PFCandidate(Kind kind, int charge, const PolarLorentzVector & p, float puppiWeight, int hwpt, int hweta, int hwphi) :
        L1Candidate(p, hwpt, hweta, hwphi, /*hwQuality=*/int(kind)),
        puppiWeight_(puppiWeight) {
    setCharge(charge);
    setPdgIdFromKind_(charge,kind);
}

void l1t::PFCandidate::setPdgIdFromKind_(int charge, Kind kind) {
    switch(kind) {
        case ChargedHadron: setPdgId(charge > 0 ? 211 : -211); break;
        case Electron: setPdgId(charge > 0 ? -11 : +11); break;
        case NeutralHadron: setPdgId(130); break;
        case Photon: setPdgId(22); break;
        case Muon: setPdgId(charge > 0 ? -13 : +13); break;
    };
}
