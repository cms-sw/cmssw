#ifndef _TOWER_PROJECTOR_H__
#define _TOWER_PROJECTOR_H__

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/LorentzVector.h"

// correcting calotower/fourvector to a new vertex
// code contributed by Ian Tomalin,
// slightly modified by Andreas Oehler


namespace reco{


  void newCaloPoint(const Particle::Vector& direction,Particle::Point& newposition);

  void physicsP4 (const Particle::Point &vertex, const Particle &inParticle, 
	     Particle::LorentzVector &returnVector);

}

#endif
