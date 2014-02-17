#ifndef PhysicsTools_JetMCAlgos_Herwig6PartonSelector_H
#define PhysicsTools_JetMCAlgos_Herwig6PartonSelector_H

/**\class Herwig6PartonSelector Herwig6PartonSelector.h PhysicsTools/JetMCAlgos/interface/Herwig6PartonSelector.h
 * \brief Herwig6 parton selector derived from the base parton selector
 */

#include "PhysicsTools/JetMCAlgos/interface/BasePartonSelector.h"


class Herwig6PartonSelector : public BasePartonSelector
{
  public:
    Herwig6PartonSelector();
    virtual ~Herwig6PartonSelector();

    void run(const edm::Handle<reco::GenParticleCollection> & particles,
             std::auto_ptr<reco::GenParticleRefVector> & partons);
};

#endif
