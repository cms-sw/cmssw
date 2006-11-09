// $Id: Booster.cc,v 1.7 2006/07/26 08:48:06 llista Exp $
#include "PhysicsTools/CandUtils/interface/Booster.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include <Math/VectorUtil.h>
using namespace std;
using namespace reco;

void Booster::set( Candidate& c ) {
  c.setP4( ROOT::Math::VectorUtil::boost( c.p4(), boost ) );
  for( Candidate::iterator d = c.begin(); d != c.end(); ++ d )
    set( * d );
}
