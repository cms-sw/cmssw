// $Id: ApplyEnergyCorrection.cc,v 1.5 2006/02/21 10:37:31 llista Exp $
#include "PhysicsTools/CandUtils/interface/ApplyEnergyCorrection.h"
using namespace reco;

ApplyEnergyCorrection::~ApplyEnergyCorrection() { }

void ApplyEnergyCorrection::set( Candidate & c ) {
  p4 = c.p4() * correction_;
}
