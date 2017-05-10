
/**
 * This is a Pythia6-specific parton selector that selects all status==2 partons. An explanation of the particle status codes
 * returned by Pythia6 can be found in Section 5.4 of Pythia6 manual (http://arxiv.org/abs/hep-ph/0603175)
 */

#include "PhysicsTools/JetMCAlgos/interface/Pythia6PartonSelector.h"
#include "PhysicsTools/JetMCUtils/interface/CandMCTag.h"


Pythia6PartonSelector::Pythia6PartonSelector()
{
}

Pythia6PartonSelector::~Pythia6PartonSelector()
{
}

void
Pythia6PartonSelector::run(const edm::Handle<reco::GenParticleCollection> & particles,
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
