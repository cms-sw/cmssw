// $Id: AddFourMomenta.cc,v 1.6 2006/07/26 08:48:06 llista Exp $
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Candidate/interface/Candidate.h"

using namespace reco;

void AddFourMomenta::set( Candidate & c ) const {
  Candidate::LorentzVector p4( 0, 0, 0, 0 );
  Candidate::Charge charge = 0;
  for( Candidate::iterator d = c.begin(); d != c.end(); ++ d ) {
    p4 += d->p4();
    charge += d->charge();
  }
  c.setP4( p4 );
  c.setCharge( charge );
}
