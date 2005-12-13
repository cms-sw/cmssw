// $Id: AddFourMomenta.cc,v 1.3 2005/12/05 14:44:53 llista Exp $
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
using namespace aod;

AddFourMomenta::~AddFourMomenta() { }

void AddFourMomenta::set( Candidate & c ) {
  p4.SetXYZT( 0, 0, 0, 0 );
  charge = 0;
  for( Candidate::iterator d = c.begin(); d != c.end(); ++ d ) {
    p4 += d->p4();
    charge += d->charge();
  }
}
