#ifndef PHYSICSTOOLS_BOOSTER_H
#define PHYSICSTOOLS_BOOSTER_H
#include "PhysicsTools/Candidate/interface/CompositeCandidate.h"

struct Booster : public aod::Candidate::setup {
  typedef aod::Candidate::Vector Vector;
  Booster( const Vector & b ) : 
    aod::Candidate::setup( setupCharge( false ), setupP4( true ) ), 
    boost( b ) { }
  virtual ~Booster();
  virtual void set( aod::Candidate& c );
  const Vector & boostVector() { return boost; }
private:
  const Vector boost;
};

struct CenterOfMassBooster : public Booster {
  CenterOfMassBooster( const aod::Candidate & c ) : Booster( c.boostToCM() ) {
  }
};

#endif
