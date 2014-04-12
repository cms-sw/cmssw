#include "PhysicsTools/CandUtils/interface/ApplyEnergyCorrection.h"
#include "DataFormats/Candidate/interface/Candidate.h"

using namespace reco;

void ApplyEnergyCorrection::set( Candidate & c ) {
  c.setP4( c.p4() * correction_ );
}
