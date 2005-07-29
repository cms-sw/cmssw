#ifndef PHYSICSTOOLS_ADDFOURMOMENTA_H
#define PHYSICSTOOLS_ADDFOURMOMENTA_H
// $Id$
#include "PhysicsTools/Candidate/interface/CompositeCandidate.h"
#include "PhysicsTools/Candidate/interface/Setup.h"

namespace phystools {

  struct AddFourMomenta : public Setup {
    AddFourMomenta() : Setup( setupCharge( true ), setupP4( true ) ) { }
    virtual ~AddFourMomenta();
    void setup( Candidate& c );
  };
  
}

#endif
