// $Id: Booster.cc,v 1.1 2005/07/29 07:05:57 llista Exp $
#include "PhysicsTools/CandUtils/interface/Booster.h"
#include "PhysicsTools/Candidate/interface/daughter_iterator.h"
using namespace std;
using namespace aod;

Booster::~Booster() { }

void Booster::set( Candidate& c ) {
  p4 = c.p4(); 
  p4.boost( boost );

  for( Candidate::iterator d = c.begin(); d != c.end(); ++ d ) {
    Booster clone( * this );
    d->set( clone );
  }
}
