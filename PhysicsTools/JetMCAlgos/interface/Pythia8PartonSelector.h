#ifndef PhysicsTools_JetMCAlgos_Pythia8PartonSelector_H
#define PhysicsTools_JetMCAlgos_Pythia8PartonSelector_H

/**\class Pythia8PartonSelector Pythia8PartonSelector.h PhysicsTools/JetMCAlgos/interface/Pythia8PartonSelector.h
 * \brief Pythia8 parton selector derived from the base parton selector
 */

#include "PhysicsTools/JetMCAlgos/interface/BasePartonSelector.h"


class Pythia8PartonSelector : public BasePartonSelector
{
  public:
    Pythia8PartonSelector();
    virtual ~Pythia8PartonSelector();

    void run(const edm::Handle<reco::GenParticleCollection> & particles,
             std::auto_ptr<reco::GenParticleRefVector> & partons);
};

#endif
