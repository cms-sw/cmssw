
/**
 * This is a Sherpa-specific parton selector that selects all status==11 partons.
 */

#include "PhysicsTools/JetMCAlgos/interface/SherpaPartonSelector.h"
#include "PhysicsTools/JetMCUtils/interface/CandMCTag.h"


SherpaPartonSelector::SherpaPartonSelector()
{
}

SherpaPartonSelector::~SherpaPartonSelector()
{
}

void
SherpaPartonSelector::run(const edm::Handle<reco::GenParticleCollection> & particles,
                          std::auto_ptr<reco::GenParticleRefVector> & partons)
{
   // loop over particles and select partons
   for(reco::GenParticleCollection::const_iterator it = particles->begin(); it != particles->end(); ++it)
   {
     if( it->status()!=11 ) continue;                  // only accept status==11 particles
     if( !CandMCTagUtils::isParton( *it ) ) continue;  // skip particle if not a parton

     partons->push_back( reco::GenParticleRef( particles, it - particles->begin() ) );
   }

   return;
}
