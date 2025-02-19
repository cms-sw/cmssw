// $Id: ApplyEnergyCorrection.cc,v 1.2 2006/07/26 08:48:06 llista Exp $
#include "PhysicsTools/CandUtils/interface/ApplyEnergyCorrection.h"
#include "DataFormats/Candidate/interface/Candidate.h"

using namespace reco;

void ApplyEnergyCorrection::set( Candidate & c ) {
  c.setP4( c.p4() * correction_ );
}
