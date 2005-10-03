// $Id: TwoDifferentBodyCombiner.cc,v 1.1 2005/10/03 09:17:45 llista Exp $
#include "PhysicsTools/CandUtils/interface/TwoDifferentBodyCombiner.h"
using namespace aod;
using namespace std;

TwoDifferentBodyCombiner::TwoDifferentBodyCombiner( double massMin, double massMax, 
						    bool ck, int q ) :
  TwoBodyCombiner( massMin, massMax, ck, q ) {
}

auto_ptr<TwoBodyCombiner::Candidates> 
TwoDifferentBodyCombiner::combine( const Candidates & cands1, 
				   const Candidates & cands2 ) {
  auto_ptr<Candidates> comps( new Candidates );
  const int n1 = cands1.size(), n2 = cands2.size();
  for( int i1 = 0; i1 < n1; ++ i1 ) {
    const Candidate & c1 = * cands1[ i1 ];
    for ( int i2 = 0; i2 < n2; ++ i2 ) {
      const Candidate & c2 = * cands2[ i2 ];
      if ( select( c1, c2 ) ) {
	comps->push_back( TwoBodyCombiner::combine( c1, c2 ) );
      }
    }
  }
  return comps;
}
