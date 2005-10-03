#include "PhysicsTools/CandUtils/interface/TwoSameBodyCombiner.h"
using namespace aod;
using namespace std;

TwoSameBodyCombiner::TwoSameBodyCombiner( double massMin, double massMax, 
					  bool ck, int q ) :
  TwoBodyCombiner( massMin, massMax, ck, q ) {
}

auto_ptr<TwoBodyCombiner::Candidates> 
TwoSameBodyCombiner::combine( const Candidates & cands ) {
  auto_ptr<Candidates> comps( new Candidates );
  const int n = cands.size();
  for( int i1 = 0; i1 < n; ++ i1 ) {
    const Candidate & c1 = * cands[ i1 ];
    for ( int i2 = i1 + 1; i2 < n; ++ i2 ) {
      const Candidate & c2 = * cands[ i2 ];
      if ( select( c1, c2 ) ) {
	comps->push_back( TwoBodyCombiner::combine( c1, c2 ) );
      }
    }
  }
  return comps;
}
