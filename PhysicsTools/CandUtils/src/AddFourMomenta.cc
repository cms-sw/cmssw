// $Id: AddFourMomenta.cc,v 1.8 2007/09/27 13:55:04 llista Exp $
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Candidate/interface/Candidate.h"

using namespace reco;

void AddFourMomenta::set( Candidate & c ) const {
  Candidate::LorentzVector p4( 0, 0, 0, 0 );
  Candidate::Charge charge = 0;
  Candidate::iterator b = c.begin(), e = c.end(); 
  for(  Candidate::iterator d = b; d != e; ++ d ) {
    p4 += d->p4();
    charge += d->charge();
  }
  c.setP4( p4 );
  c.setCharge( charge );
}
