// $Id: ApplyEnergyCorrection.cc,v 1.1 2006/07/24 06:44:17 llista Exp $
#include "PhysicsTools/CandUtils/interface/ApplyEnergyCorrection.h"
#include "DataFormats/Candidate/interface/Candidate.h"

using namespace reco;

void ApplyEnergyCorrection::set( Candidate & c ) {
  c.setP4( c.p4() * correction_ );
}
