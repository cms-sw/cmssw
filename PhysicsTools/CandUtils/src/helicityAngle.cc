// $Id: helicityAngle.cc,v 1.14 2007/12/11 08:42:53 hegner Exp $
#include "PhysicsTools/CandUtils/interface/helicityAngle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/CandUtils/interface/Booster.h"
#include <Math/VectorUtil.h>
using namespace reco;
using namespace std;

double helicityAngle( const reco::Candidate & mother, const reco::Candidate & daughter) {
  Particle::Vector boost = mother.p4().BoostToCM();
  Particle::LorentzVector pdau = ROOT::Math::VectorUtil::boost( daughter.p4(), boost );
  double h = ROOT::Math::VectorUtil::Angle( pdau, boost );
  if ( h > M_PI / 2 ) h = M_PI - h;
  return h;  
}


double helicityAngle( const Candidate & cand ) {
  assert( cand.numberOfDaughters() == 2 );
  return helicityAngle( cand, *cand.daughter(0) );
}


