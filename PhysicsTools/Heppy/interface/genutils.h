#ifndef PhysicsTools_Heppy_genutils_h
#define PhysicsTools_Heppy_genutils_h

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

namespace heppy {
    class GenParticleRefHelper {
        public:
            static int motherKey(const reco::GenParticle &gp, int index)  ;
            static int daughterKey(const reco::GenParticle &gp, int index)  ;
    };
}
#endif
