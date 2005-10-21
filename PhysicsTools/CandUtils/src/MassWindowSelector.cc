#include "PhysicsTools/CandUtils/interface/MassWindowSelector.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"

bool MassWindowSelector::operator()( const aod::Candidate & c ) const {
  double m2 = c.massSqr();
  return mMin2 < m2 && m2 < mMax2;
}
