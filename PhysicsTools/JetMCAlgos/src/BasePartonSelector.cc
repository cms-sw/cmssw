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

bool
BasePartonSelector::isParton(const reco::Candidate* const cand)
{
   int id = abs(cand->pdgId());

   if( id == 1 ||
       id == 2 ||
       id == 3 ||
       id == 4 ||
       id == 5 ||
       id == 21 )
     return true;
   else
     return false;
}
