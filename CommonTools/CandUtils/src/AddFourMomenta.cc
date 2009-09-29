// $Id: AddFourMomenta.cc,v 1.1 2009/02/26 09:17:34 llista Exp $
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Candidate/interface/Candidate.h"

using namespace reco;

void AddFourMomenta::set( Candidate & c ) const {
  Candidate::LorentzVector p4( 0, 0, 0, 0 );
  Candidate::Charge charge = 0;
  Candidate::const_iterator b = c.begin(), e = c.end(); 
  for(  Candidate::const_iterator d = b; d != e; ++ d ) {
    p4 += d->p4();
    charge += d->charge();
  }
  c.setP4( p4 );
  c.setCharge( charge );
}
