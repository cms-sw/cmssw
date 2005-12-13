// $Id: helicityAngle.cc,v 1.6 2005/12/05 14:44:53 llista Exp $
#include "PhysicsTools/CandUtils/interface/helicityAngle.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "PhysicsTools/CandUtils/interface/Booster.h"
using namespace aod;
using namespace std;

double helicityAngle( const Candidate & cand ) {
  assert( cand.numberOfDaughters() == 2 );
  Particle::Vector boost = - cand.p4().BoostVector(); 
  Particle::LorentzVector dp = cand.daughter( 0 ).p4();
  dp.Boost( boost );
  double h = dp.Angle( boost );
  if ( h > M_PI / 2 ) h = M_PI - h;
  return h;
}
