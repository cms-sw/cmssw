// $Id: helicityAngle.cc,v 1.1 2005/07/29 07:05:57 llista Exp $
#include "PhysicsTools/CandUtils/interface/helicityAngle.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "PhysicsTools/CandUtils/interface/Booster.h"
using namespace aod;
using namespace std;

double helicityAngle( const Candidate & cand ) {
  assert( cand.numberOfDaughters() == 2 );
  Particle::Vector boost = - cand.p4().boostVector(); 
  Particle::LorentzVector dp = cand[ 0 ]->p4();
  dp.boost( boost );
  double h = dp.angle( boost );
  if ( h > M_PI / 2 ) h = M_PI - h;
  return h;
}
