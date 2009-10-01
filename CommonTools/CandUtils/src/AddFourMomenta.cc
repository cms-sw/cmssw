// $Id: AddFourMomenta.cc,v 1.2 2009/09/29 12:24:45 llista Exp $
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Candidate/interface/Candidate.h"
using namespace reco;

void AddFourMomenta::set( Candidate & c ) const {
  Candidate::LorentzVector p4( 0, 0, 0, 0 );
  Candidate::Charge charge = 0;
  size_t n = c.numberOfDaughters();
  for(size_t i = 0; i < n; ++i) {
    const Candidate * d = (const_cast<const Candidate &>(c)).daughter(i);
    p4 += d->p4();
    charge += d->charge();
  }
  c.setP4( p4 );
  c.setCharge( charge );
}
