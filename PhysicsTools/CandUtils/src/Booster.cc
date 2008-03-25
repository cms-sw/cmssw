// $Id: Booster.cc,v 1.8 2006/11/09 09:24:51 llista Exp $
#include "PhysicsTools/CandUtils/interface/Booster.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include <Math/VectorUtil.h>
using namespace std;
using namespace reco;

void Booster::set( Candidate& c ) {
  c.setP4( ROOT::Math::VectorUtil::boost( c.p4(), boost ) );
  Candidate::iterator b = c.begin(), e = c.end(); 
  for(  Candidate::iterator d = b; d != e; ++ d )
    set( * d );
}
