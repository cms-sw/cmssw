#include "DataFormats/Phase2L1ParticleFlow/interface/PFJet.h"

void l1t::PFJet::calibratePt(float newpt) {
    setP4(PolarLorentzVector(newpt, eta(), phi(), mass()));
}


