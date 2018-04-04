#include "DataFormats/Phase2L1ParticleFlow/interface/PFCandidate.h"

void l1t::PFCandidate::setPdgIdFromKind_(int charge, Kind kind) {
    switch(kind) {
        case ChargedHadron: setPdgId(charge ? 211 : -211); break;
        case Electron: setPdgId(charge ? -11 : +11); break;
        case NeutralHadron: setPdgId(130); break;
        case Photon: setPdgId(22); break;
        case Muon: setPdgId(charge ? -13 : +13); break;
    };
}
