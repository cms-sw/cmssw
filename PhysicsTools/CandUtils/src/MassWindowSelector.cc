// $Id$
#include "PhysicsTools/CandUtils/interface/MassWindowSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

bool MassWindowSelector::operator()( const reco::Candidate & c ) const {
  double m2 = c.massSqr();
  return mMin2 < m2 && m2 < mMax2;
}
