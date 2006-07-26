// $Id: helicityAngle.cc,v 1.10 2006/02/28 11:20:21 llista Exp $
#include "PhysicsTools/CandUtils/interface/helicityAngle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/CandUtils/interface/Booster.h"
#include <Math/VectorUtil.h>
using namespace reco;
using namespace std;

double helicityAngle( const Candidate & cand ) {
  assert( cand.numberOfDaughters() == 2 );
  Particle::Vector boost = cand.p4().BoostToCM(); 
  Particle::LorentzVector pdau = ROOT::Math::VectorUtil::Boost( cand.daughter( 0 ).p4(), boost );
  double h = ROOT::Math::VectorUtil::Angle( pdau, boost );
  if ( h > M_PI / 2 ) h = M_PI - h;
  return h;
}
