
/**
 * This is a Pythia8-specific parton selector that selects all status==43, 44, 51, 52, and 62 partons. An explanation of
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
     if( !(status==43 || status==44 || status==51 || status==52 || status==62) ) continue; // only accept status==43, 44, 51, 52, and 62 particles
     if( it->numberOfDaughters()==0 ) continue;                                            // skip particle if it has no daughters (likely a documentation line)
     if( !CandMCTagUtils::isParton( *it ) ) continue;                                      // skip particle if not a parton

     partons->push_back( reco::GenParticleRef( particles, it - particles->begin() ) );
   }

   return;
}
