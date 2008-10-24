#include "PhysicsTools/StarterKit/interface/HistoParticle.h"

#include <iostream>

using namespace pat;
using namespace std;

// Constructor:

HistoParticle::HistoParticle(std::string dir, std::string group,std::string pre,
			     double pt1, double pt2, double m1, double m2,
			     TFileDirectory * parentDir)
  : HistoGroup<Particle>( dir, group, pre, pt1, pt2, m1, m2, parentDir)
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

void HistoParticle::fill( const reco::ShallowClonePtrCandidate * pshallow, double weight )
{


  // Get the underlying object that the shallow clone represents
  const pat::Particle * particle = dynamic_cast<const pat::Particle*>(&*(pshallow->masterClonePtr()));

  if ( particle == 0 ) {
    cout << "Error! Was passed a shallow clone that is not at heart a particle" << endl;
    return;
  }


  // First fill common 4-vector histograms
  HistoGroup<Particle>::fill( pshallow, 1, weight );

  // fill relevant particle histograms
}


void HistoParticle::clearVec()
{
  HistoGroup<Particle>::clearVec();
}
