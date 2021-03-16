#ifndef MatchedParton_H
#define MatchedParton_H

#include <vector>
//#include "DataFormats/Candidate/interface/Candidate.h"
//#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

namespace reco {

  class MatchedPartons {
  public:
    MatchedPartons() {}
    MatchedPartons(GenParticleRef hv, GenParticleRef n2, GenParticleRef n3, GenParticleRef pd, GenParticleRef ad)
        : m_heaviest(hv), m_nearest2(n2), m_nearest3(n3), m_PhysDef(pd), m_AlgoDef(ad) {}

    //Return the ParticleRef for the heaviest flavour in the signal cone
    const GenParticleRef heaviest() const { return m_heaviest; }

    //Return the ParticleRef for the nearest parton (status=2)
    const GenParticleRef& nearest_status2() const { return m_nearest2; }

    //Return the ParticleRef for the nearest parton (status=3)
    const GenParticleRef& nearest_status3() const { return m_nearest3; }

    //Return the ParticleRef for the Physics Definition parton
    const GenParticleRef& physicsDefinitionParton() const { return m_PhysDef; }

    //Return the ParticleRef for the Algorithmic Definition parton
    const GenParticleRef& algoDefinitionParton() const { return m_AlgoDef; }

  private:
    GenParticleRef m_heaviest;
    GenParticleRef m_nearest2;
    GenParticleRef m_nearest3;
    GenParticleRef m_PhysDef;
    GenParticleRef m_AlgoDef;
  };

}  // namespace reco
#endif
