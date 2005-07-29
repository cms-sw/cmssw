#ifndef PHYSICSTOOLS_BOOSTER_H
#define PHYSICSTOOLS_BOOSTER_H
#include "PhysicsTools/Candidate/interface/CompositeCandidate.h"
#include "PhysicsTools/Candidate/interface/Setup.h"
#include <CLHEP/Vector/ThreeVector.h>

namespace phystools {

  struct Booster : public Setup {
    Booster( const CLHEP::Hep3Vector & b ) : 
      Setup( setupCharge( false ), setupP4( true ) ), 
      boost( b ) { }
    virtual ~Booster();
    virtual void setup( Candidate& c );
    const CLHEP::Hep3Vector & boostVector() { return boost; }
  private:
    const CLHEP::Hep3Vector boost;
  };
  
  struct CenterOfMassBooster : public Booster {
    CenterOfMassBooster( const Candidate & c ) : Booster( - c.p4().boostVector() ) {
    }
  };

}

#endif
