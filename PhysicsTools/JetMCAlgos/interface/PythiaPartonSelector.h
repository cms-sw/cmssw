#ifndef PhysicsTools_JetMCAlgos_PythiaPartonSelector_H
#define PhysicsTools_JetMCAlgos_PythiaPartonSelector_H

/**\class PythiaPartonSelector PythiaPartonSelector.h PhysicsTools/JetMCAlgos/interface/PythiaPartonSelector.h
 * \brief Pythia parton selector derived from the base parton selector
 */

#include "PhysicsTools/JetMCAlgos/interface/BasePartonSelector.h"


class PythiaPartonSelector : public BasePartonSelector
{
  public:
    PythiaPartonSelector();
    ~PythiaPartonSelector();

    void run(const edm::Handle<reco::GenParticleCollection> & particles,
             std::auto_ptr<reco::GenParticleRefVector> & partons);
};

#endif
