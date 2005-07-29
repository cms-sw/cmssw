// $Id$

#include "PhysicsTools/CandUtils/interface/Booster.h"
#include "PhysicsTools/Candidate/interface/daughter_iterator.h"
#include <iostream>

using namespace std;
using namespace phystools;

Booster::~Booster() { }

void Booster::setup( Candidate& c ) {
  p4 = c.p4(); 
  p4.boost( boost );

  for( Candidate::iterator d = c.begin(); d != c.end(); ++ d ) {
    Booster clone( * this );
    d->setup( clone );
  }
}
