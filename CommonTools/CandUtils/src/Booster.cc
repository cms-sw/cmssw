// $Id: Booster.cc,v 1.9 2007/09/27 13:55:04 llista Exp $
#include "CommonTools/CandUtils/interface/Booster.h"
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
