// $Id$
#include "PhysicsTools/CandUtils/interface/PtMinSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

bool PtMinSelector::operator()( const reco::Candidate & c ) const {
  return c.pt() > ptMin;
}
