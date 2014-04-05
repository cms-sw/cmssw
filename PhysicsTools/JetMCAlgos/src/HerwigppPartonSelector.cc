
/**
 * This is a Herwig++-specific parton selector that selects all status==2 partons. This is likely a temporary choice since
 * Herwig++ status codes in CMSSW currently break the HepMC convention. Once the status codes are fixed, the selector will
 * be updated.
 */

#include "PhysicsTools/JetMCAlgos/interface/HerwigppPartonSelector.h"
#include "PhysicsTools/JetMCUtils/interface/CandMCTag.h"


HerwigppPartonSelector::HerwigppPartonSelector()
{
}

HerwigppPartonSelector::~HerwigppPartonSelector()
{
}

void
HerwigppPartonSelector::run(const edm::Handle<reco::GenParticleCollection> & particles,
                          std::auto_ptr<reco::GenParticleRefVector> & partons)
{
   // loop over particles and select partons
   for(reco::GenParticleCollection::const_iterator it = particles->begin(); it != particles->end(); ++it)
   {
     if( it->status()!=2 ) continue;                   // only accept status==2 particles
     if( !CandMCTagUtils::isParton( *it ) ) continue;  // skip particle if not a parton

     partons->push_back( reco::GenParticleRef( particles, it - particles->begin() ) );
   }

   return;
}
