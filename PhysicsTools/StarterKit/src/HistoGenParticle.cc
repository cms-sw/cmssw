#include "PhysicsTools/StarterKit/interface/HistoGenParticle.h"

#include <iostream>

using pat::HistoGenParticle;
using namespace std;

// Constructor:

HistoGenParticle::HistoGenParticle(std::string dir, std::string group,std::string pre,
				   double pt1, double pt2, double m1, double m2,
				   TFileDirectory * parentDir)
  : HistoGroup<reco::GenParticle>( dir, group, pre, pt1, pt2, m1, m2, parentDir, true)
{
  // book relevant particle histograms
}

HistoGenParticle::~HistoGenParticle()
{
  // Root deletes histograms, not us
}


void HistoGenParticle::fill( const reco::GenParticle * particle, double weight )
{

  // First fill common 4-vector histograms
  int key = getKey( particle );
  if ( key != 9999 )
    HistoGroup<reco::GenParticle>::fill( particle, key, weight );

  // fill relevant particle histograms
}

void HistoGenParticle::fill( const reco::ShallowClonePtrCandidate * pshallow, double weight )
{


  // Get the underlying object that the shallow clone represents
  const reco::GenParticle * particle = dynamic_cast<const reco::GenParticle*>(&*(pshallow->masterClonePtr()));

  if ( particle == 0 ) {
    cout << "Error! Was passed a shallow clone that is not at heart a particle" << endl;
    return;
  }


  // First fill common 4-vector histograms
  int key = getKey( pshallow );
  if ( key != 9999 )
    HistoGroup<reco::GenParticle>::fill( pshallow, key, weight );

  // fill relevant particle histograms
}


void HistoGenParticle::fillCollection( const std::vector<reco::GenParticle> & coll, double weight )
{

  h_size_->fill( coll.size(), 1, weight );     //! Save the size of the collection.

  std::vector<reco::GenParticle>::const_iterator
    iobj = coll.begin(),
    iend = coll.end();

  for ( ; iobj != iend; ++iobj ) {
    fill( &*iobj, weight);      //! &*iobj dereferences to the pointer to a PHYS_OBJ*
  }
}



void HistoGenParticle::clearVec()
{
  HistoGroup<reco::GenParticle>::clearVec();
}


int HistoGenParticle::getKey( reco::Candidate const * p ) 
{
  int status = p->status();
  int pdgId = abs(p->pdgId());

  // First look at status 3 particles: only hard scatter
  if ( status == 3 ) {
    // Basic quarks, leptons, bosons
    if ( (pdgId >= 1  && pdgId <= 8  ) ||    // quarks
// 	 (pdgId >= 11 && pdgId <= 18 ) ||    // leptons
	 (pdgId >= 21 && pdgId <= 37 )       // bosons
	 ) return pdgId;
    // Otherwise ignore it
    else return 9999;
  }

  // Next look at status 2 particles: only decays, otherwise flag them as "status 2"
  else if ( status == 2 ) {
    // photons
    if ( pdgId == 22 ) return status * 10000 + pdgId;
    // taus
    if ( pdgId == 15 ) return pdgId;
    // baryons 
    if ( pdgId / 1000 > 0 && pdgId / 1000 < 10 ) return pdgId % 10000;
    // mesons
    if ( pdgId / 1000 == 0 ) return pdgId % 1000;
    // Otherwise ignore it
    return 9999;
  }

  // Finally look at status 3 particles: stable particles
  else if ( status == 1 ) {
    // leptons (aside from taus, handled in status 2)
    if ( pdgId >= 11 && pdgId <= 18 && pdgId != 15 ) return pdgId;
    // photons
    if ( pdgId == 22 ) return pdgId;
    // baryons 
    if ( pdgId / 1000 > 0 && pdgId / 1000 < 10 ) return pdgId % 10000;
    // mesons
    if ( pdgId / 1000 == 0 ) return pdgId % 1000;
    // Otherwise ignore it
    return 9999;
  }

  return 9999;
}
