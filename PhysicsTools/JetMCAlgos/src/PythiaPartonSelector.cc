
/**
 * This is a Pythia-specific parton selector which selects partons from the end of the parton showering sequence, i.e.,
 * it selects partons that do not have other partons as daughters. It works with both Pythia6 and Pythia8.
 */

#include "PhysicsTools/JetMCAlgos/interface/PythiaPartonSelector.h"


PythiaPartonSelector::PythiaPartonSelector()
{
}

PythiaPartonSelector::~PythiaPartonSelector()
{
}

void
PythiaPartonSelector::run(const edm::Handle<reco::GenParticleCollection> & particles,
                          std::auto_ptr<reco::GenParticleRefVector> & partons)
{
   // loop over particles and select partons
   for(reco::GenParticleCollection::const_iterator it = particles->begin(); it != particles->end(); ++it)
   {
     if( !isParton( &(*it) ) ) continue;        // skip particle if not parton
     if( it->numberOfDaughters()==0 ) continue; // skip particle if it has no daughters

     // check if any of the daughters is also a parton
     bool hasPartonDaughter = false;
     for(size_t i=0; i < it->numberOfDaughters(); ++i)
     {
       if( isParton( it->daughter(i) ) ) { hasPartonDaughter = true; break; }
     }
     if( hasPartonDaughter ) continue; // skip partons that have other partons as daughters

     partons->push_back( reco::GenParticleRef( particles, it - particles->begin() ) );
   }

   return;
}
