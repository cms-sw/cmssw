#include "PhysicsTools/JetMCAlgos/interface/BasePartonSelector.h"

BasePartonSelector::BasePartonSelector()
{
}

BasePartonSelector::~BasePartonSelector()
{
}

void
BasePartonSelector::run(const edm::Handle<reco::GenParticleCollection> & particles,
                        std::auto_ptr<reco::GenParticleRefVector> & partons)
{
}
