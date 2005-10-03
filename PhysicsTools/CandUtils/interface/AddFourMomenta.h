#ifndef PHYSICSTOOLS_ADDFOURMOMENTA_H
#define PHYSICSTOOLS_ADDFOURMOMENTA_H
// $Id: AddFourMomenta.h,v 1.1 2005/07/29 07:05:57 llista Exp $
#include "PhysicsTools/Candidate/interface/CompositeCandidate.h"

struct AddFourMomenta : public aod::Candidate::setup {
  AddFourMomenta() : aod::Candidate::setup( setupCharge( true ), setupP4( true ) ) { }
  virtual ~AddFourMomenta();
  void set( aod::Candidate& c );
};

#endif
