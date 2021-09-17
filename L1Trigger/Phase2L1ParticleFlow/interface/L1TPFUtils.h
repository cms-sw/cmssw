#ifndef L1Trigger_Phase2L1ParticleFlow_L1TPFUtils_h
#define L1Trigger_Phase2L1ParticleFlow_L1TPFUtils_h
#include <vector>
#include "DataFormats/Math/interface/LorentzVector.h"

namespace l1tpf {
  std::pair<float, float> propagateToCalo(const math::XYZTLorentzVector& iMom,
                                          const math::XYZTLorentzVector& iVtx,
                                          double iCharge,
                                          double iBField);
}

#endif
