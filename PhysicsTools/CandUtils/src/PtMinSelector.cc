#include "PhysicsTools/CandUtils/interface/PtMinSelector.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"

bool PtMinSelector::operator()( const reco::Candidate & c ) const {
  return c.pt() > ptMin;
}
