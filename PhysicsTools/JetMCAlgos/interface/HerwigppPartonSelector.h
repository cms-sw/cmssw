#ifndef PhysicsTools_JetMCAlgos_HerwigppPartonSelector_H
#define PhysicsTools_JetMCAlgos_HerwigppPartonSelector_H

/**\class HerwigppPartonSelector HerwigppPartonSelector.h PhysicsTools/JetMCAlgos/interface/HerwigppPartonSelector.h
 * \brief Herwig++ parton selector derived from the base parton selector
 */

#include "PhysicsTools/JetMCAlgos/interface/BasePartonSelector.h"


class HerwigppPartonSelector : public BasePartonSelector
{
  public:
    HerwigppPartonSelector();
    virtual ~HerwigppPartonSelector();

    void run(const edm::Handle<reco::GenParticleCollection> & particles,
             std::auto_ptr<reco::GenParticleRefVector> & partons);
};

#endif
