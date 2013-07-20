// $Id: ApplyEnergyCorrection.cc,v 1.1 2009/02/26 09:17:34 llista Exp $
#include "CommonTools/CandUtils/interface/ApplyEnergyCorrection.h"
#include "DataFormats/Candidate/interface/Candidate.h"

using namespace reco;

void ApplyEnergyCorrection::set( Candidate & c ) {
  c.setP4( c.p4() * correction_ );
}
