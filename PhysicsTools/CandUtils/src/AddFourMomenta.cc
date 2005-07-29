// $Id$

#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "PhysicsTools/Candidate/interface/daughter_iterator.h"

using namespace phystools;

AddFourMomenta::~AddFourMomenta() { }

void AddFourMomenta::setup( Candidate & c ) {
  p4.set( 0, 0, 0, 0 );
  charge = 0;
  for( Candidate::iterator d = c.begin(); d != c.end(); ++ d ) {
    p4 += d->p4();
    charge += d->charge();
  }
}
