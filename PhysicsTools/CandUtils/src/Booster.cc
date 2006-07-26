// $Id: Booster.cc,v 1.6 2006/02/21 10:37:31 llista Exp $
#include "PhysicsTools/CandUtils/interface/Booster.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include <Math/VectorUtil.h>
using namespace std;
using namespace reco;

void Booster::set( Candidate& c ) {
  c.setP4( ROOT::Math::VectorUtil::Boost( c.p4(), boost ) );
  for( Candidate::iterator d = c.begin(); d != c.end(); ++ d )
    set( * d );
}
