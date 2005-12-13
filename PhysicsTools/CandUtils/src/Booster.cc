// $Id: Booster.cc,v 1.3 2005/12/05 14:44:53 llista Exp $
#include "PhysicsTools/CandUtils/interface/Booster.h"
using namespace std;
using namespace aod;

Booster::~Booster() { }

void Booster::set( Candidate& c ) {
  p4 = c.p4(); 
  p4.Boost( boost );

  for( Candidate::iterator d = c.begin(); d != c.end(); ++ d ) {
    Booster clone( * this );
    d->set( clone );
  }
}
