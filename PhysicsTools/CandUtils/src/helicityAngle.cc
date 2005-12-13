// $Id: helicityAngle.cc,v 1.7 2005/12/13 01:47:12 llista Exp $
#include "PhysicsTools/CandUtils/interface/helicityAngle.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "PhysicsTools/CandUtils/interface/Booster.h"
#include <Math/VectorUtil.h>
using namespace aod;
using namespace std;

double helicityAngle( const Candidate & cand ) {
  assert( cand.numberOfDaughters() == 2 );
  Particle::Vector boost = cand.p4().BoostToCM(); 
  Particle::LorentzVector dp = cand.daughter( 0 ).p4();
  //  dp.Boost( boost );

  //  p4.Boost( boost );
  // boost is not supported in ROOT:Math ??

  double bx = boost.X(), by = boost.Y(), bz = boost.Z();
  double x = dp.X(), y = dp.Y(), z = dp.Y(), t = dp.T();
  double b2 = bx * bx + by * by + bz * bz;
  register double gamma = 1.0 / sqrt(1.0 - b2);
  register double bp = bx * x + by * y + bz * z;
  register double gamma2 = b2 > 0 ? ( gamma - 1.0 ) / b2 : 0.0;
  dp.SetXYZT( x + gamma2 * bp * bx + gamma * bx * t,
	      y + gamma2 * bp * by + gamma * by * t,
	      z + gamma2 * bp * bz + gamma * bz * t,
	      gamma * ( t + bp ) );

  double h = ROOT::Math::VectorUtil::Angle( dp, boost );
  if ( h > M_PI / 2 ) h = M_PI - h;
  return h;
}
