// $Id: Booster.cc,v 1.2 2005/10/03 10:12:11 llista Exp $
#include "PhysicsTools/CandUtils/interface/Booster.h"
#include "PhysicsTools/Candidate/interface/daughter_iterator.h"
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
