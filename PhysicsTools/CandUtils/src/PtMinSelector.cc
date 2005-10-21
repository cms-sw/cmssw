#include "PhysicsTools/CandUtils/interface/PtMinSelector.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"

bool PtMinSelector::operator()( const aod::Candidate & c ) const {
  return c.pt() > ptMin;
}
