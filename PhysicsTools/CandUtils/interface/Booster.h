#ifndef PHYSICSTOOLS_BOOSTER_H
#define PHYSICSTOOLS_BOOSTER_H
#include "PhysicsTools/Candidate/interface/CompositeCandidate.h"
#include <CLHEP/Vector/ThreeVector.h>

struct Booster : public aod::Candidate::setup {
  Booster( const CLHEP::Hep3Vector & b ) : 
    aod::Candidate::setup( setupCharge( false ), setupP4( true ) ), 
    boost( b ) { }
  virtual ~Booster();
  virtual void set( aod::Candidate& c );
  const CLHEP::Hep3Vector & boostVector() { return boost; }
private:
  const CLHEP::Hep3Vector boost;
};

struct CenterOfMassBooster : public Booster {
  CenterOfMassBooster( const aod::Candidate & c ) : Booster( - c.p4().boostVector() ) {
  }
};

#endif
