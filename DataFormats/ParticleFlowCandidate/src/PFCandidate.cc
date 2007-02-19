#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include <iostream>

using namespace reco;


PFCandidate * PFCandidate::clone() const {
  return new PFCandidate( * this );
}


