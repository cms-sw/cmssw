
/**
 * This is a Pythia8-specific parton selector that selects all status==71 or 72 partons. An explanation of
 * the particle status codes returned by Pythia8 can be found in Pythia8 online manual
 * (http://home.thep.lu.se/~torbjorn/pythia81html/ParticleProperties.html).
 */

#include "PhysicsTools/JetMCAlgos/interface/Pythia8PartonSelector.h"
#include "PhysicsTools/JetMCUtils/interface/CandMCTag.h"


Pythia8PartonSelector::Pythia8PartonSelector()
{
}

Pythia8PartonSelector::~Pythia8PartonSelector()
{
}

void
Pythia8PartonSelector::run(const edm::Handle<reco::GenParticleCollection> & particles,
                          std::auto_ptr<reco::GenParticleRefVector> & partons)
{
   // loop over particles and select partons
   for(reco::GenParticleCollection::const_iterator it = particles->begin(); it != particles->end(); ++it)
   {
     int status = it->status();
     if( !(status==71 || status==72) ) continue;       // only accept status==71 or 72 particles
     if( !CandMCTagUtils::isParton( *it ) ) continue;  // skip particle if not a parton

     partons->push_back( reco::GenParticleRef( particles, it - particles->begin() ) );
   }

   return;
}
