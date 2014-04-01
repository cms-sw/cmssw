#ifndef PhysicsTools_JetMCAlgos_Pythia6PartonSelector_H
#define PhysicsTools_JetMCAlgos_Pythia6PartonSelector_H

/**\class Pythia6PartonSelector Pythia6PartonSelector.h PhysicsTools/JetMCAlgos/interface/Pythia6PartonSelector.h
 * \brief Pythia6 parton selector derived from the base parton selector
 */

#include "PhysicsTools/JetMCAlgos/interface/BasePartonSelector.h"


class Pythia6PartonSelector : public BasePartonSelector
{
  public:
    Pythia6PartonSelector();
    virtual ~Pythia6PartonSelector();

    void run(const edm::Handle<reco::GenParticleCollection> & particles,
             std::auto_ptr<reco::GenParticleRefVector> & partons);
};

#endif
