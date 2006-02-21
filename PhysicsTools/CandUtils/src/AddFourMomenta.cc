// $Id: AddFourMomenta.cc,v 1.4 2005/12/13 01:47:12 llista Exp $
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
using namespace reco;

AddFourMomenta::~AddFourMomenta() { }

void AddFourMomenta::set( Candidate & c ) {
  p4.SetXYZT( 0, 0, 0, 0 );
  charge = 0;
  for( Candidate::iterator d = c.begin(); d != c.end(); ++ d ) {
    p4 += d->p4();
    charge += d->charge();
  }
}
