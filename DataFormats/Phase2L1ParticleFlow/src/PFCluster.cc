#include "DataFormats/Phase2L1ParticleFlow/interface/PFCluster.h"

void l1t::PFCluster::calibratePt(float newpt, float preserveEmEt) {
    float currEmEt = emEt();
    ptError_ *= newpt / pt();
    setP4(PolarLorentzVector(newpt, eta(), phi(), mass()));
    if (preserveEmEt) {
        float hNew = std::max<float>(pt() - currEmEt, 0);
        hOverE_ = (currEmEt > 0 ? hNew/currEmEt : -1);
    }
}


