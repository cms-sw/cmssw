#ifndef PhysicsTools_JetMCAlgos_BasePartonSelector_H
#define PhysicsTools_JetMCAlgos_BasePartonSelector_H

/**\class BasePartonSelector BasePartonSelector.h PhysicsTools/JetMCAlgos/interface/BasePartonSelector.h
 * \brief Base parton selector from which all other generator-specific parton selectors are derived
 */

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"


class BasePartonSelector
{
  public:
    BasePartonSelector();
    ~BasePartonSelector();

    virtual void run(const edm::Handle<reco::GenParticleCollection> & particles,
                     std::auto_ptr<reco::GenParticleRefVector> & partons);
};

#endif
