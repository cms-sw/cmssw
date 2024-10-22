#include "DataFormats/L1TParticleFlow/interface/PFCluster.h"

void l1t::PFCluster::calibratePt(float newpt, float preserveEmEt) {
  float currEmEt = emEt();
  ptError_ *= newpt / pt();
  setP4(PolarLorentzVector(newpt, eta(), phi(), mass()));
  if (preserveEmEt) {
    float hNew = pt() - currEmEt;
    hOverE_ = (currEmEt > 0 ? hNew / currEmEt : -1);
  }
}
