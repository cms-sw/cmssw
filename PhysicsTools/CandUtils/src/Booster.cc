// $Id: Booster.cc,v 1.5 2005/12/13 23:27:22 llista Exp $
#include "PhysicsTools/CandUtils/interface/Booster.h"
#include <Math/LorentzRotation.h>
using namespace std;
using namespace reco;

Booster::~Booster() { }

void Booster::set( Candidate& c ) {
  p4 = c.p4(); 

  //  p4.Boost( boost );
  // boost is not supported in ROOT:Math ??

  double bx = boost.X(), by = boost.Y(), bz = boost.Z();
  double x = p4.X(), y = p4.Y(), z = p4.Y(), t = p4.T();
  double b2 = bx * bx + by * by + bz * bz;
  register double gamma = 1.0 / sqrt(1.0 - b2);
  register double bp = bx * x + by * y + bz * z;
  register double gamma2 = b2 > 0 ? ( gamma - 1.0 ) / b2 : 0.0;
  p4.SetXYZT( x + gamma2 * bp * bx + gamma * bx * t,
	      y + gamma2 * bp * by + gamma * by * t,
	      z + gamma2 * bp * bz + gamma * bz * t,
	      gamma * ( t + bp ) );

  for( Candidate::iterator d = c.begin(); d != c.end(); ++ d ) {
    Booster clone( * this );
    d->set( clone );
  }
}
