#include "PhysicsTools/StarterKit/interface/HistoParticle.h"


using pat::HistoParticle;

// Constructor:

HistoParticle::HistoParticle( std::string subDir ) :
  HistoGroup<Particle>( subDir.append("particle") )
{


  // book relevant particle histograms
}

HistoParticle::~HistoParticle()
{
  // Root deletes histograms, not us
}


void HistoParticle::fill( const Particle * particle, double weight )
{

  // First fill common 4-vector histograms
  HistoGroup<Particle>::fill( particle, 1, weight );

  // fill relevant particle histograms
}


void HistoParticle::clearVec()
{
  HistoGroup<Particle>::clearVec();
}
