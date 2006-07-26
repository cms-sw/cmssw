// $Id: AddFourMomenta.cc,v 1.5 2006/02/21 10:37:31 llista Exp $
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Candidate/interface/Candidate.h"

using namespace reco;

void AddFourMomenta::set( Candidate & c ) {
  Candidate::LorentzVector p4( 0, 0, 0, 0 );
  Candidate::Charge charge = 0;
  for( Candidate::iterator d = c.begin(); d != c.end(); ++ d ) {
    p4 += d->p4();
    charge += d->charge();
  }
  c.setP4( p4 );
  c.setCharge( charge );
}
