#ifndef PHYSICSTOOLS_ADDFOURMOMENTA_H
#define PHYSICSTOOLS_ADDFOURMOMENTA_H
// $Id: AddFourMomenta.h,v 1.2 2005/10/03 10:12:11 llista Exp $
#include "PhysicsTools/Candidate/interface/CompositeCandidate.h"

struct AddFourMomenta : public reco::Candidate::setup {
  AddFourMomenta() : reco::Candidate::setup( setupCharge( true ), setupP4( true ) ) { }
  virtual ~AddFourMomenta();
  void set( reco::Candidate& c );
};

#endif
