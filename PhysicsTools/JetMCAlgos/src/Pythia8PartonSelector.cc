
/**
 * This is a Pythia8-specific parton selector that selects all partons that don't have other partons as daughters, i.e., partons
 * from the end of the parton showering sequence. An explanation of the particle status codes returned by Pythia8 can be found in
 * Pythia8 online manual (http://home.thep.lu.se/~torbjorn/pythia81html/ParticleProperties.html).
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
     if( status==1 ) continue;                         // skip stable particles
     if( status==2 ) continue;                         // skip decayed Standard Model hadrons and leptons
     if( !CandMCTagUtils::isParton( *it ) ) continue;  // skip particle if not a parton

     // check if the parton has other partons as daughters
     int nparton_daughters = 0;
     for(unsigned i=0; i<it->numberOfDaughters(); ++i)
     {
       if( CandMCTagUtils::isParton( *(it->daughter(i)) ) )
         ++nparton_daughters;
     }

     if( nparton_daughters==0 )
       partons->push_back( reco::GenParticleRef( particles, it - particles->begin() ) );
   }

   return;
}
