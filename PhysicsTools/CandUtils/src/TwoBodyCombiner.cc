// $Id: TwoBodyCombiner.cc,v 1.2 2005/10/03 10:12:11 llista Exp $
#include "PhysicsTools/CandUtils/interface/TwoBodyCombiner.h"
using namespace aod;
using namespace std;

TwoBodyCombiner::TwoBodyCombiner( double massMin, double massMax, 
				  bool ck, int q ) :
  mass2min( massMin ), mass2max( massMax ),
  checkCharge( ck ), charge( 0 ),
  overlap() {
  if ( checkCharge ) charge = q;
  mass2min *= mass2min;
  mass2max *= mass2max;
}
	 
bool TwoBodyCombiner::select( const Candidate & c1, const Candidate & c2 ) const {
  if ( checkCharge ) {
    int q = c1.charge() + c2.charge();
    if ( q != charge ) return false;
  }
  double mass2 = ( c1.p4() + c2.p4() ).mag2();
  if ( mass2min > mass2 || mass2 > mass2max ) return false;
  if ( overlap( c1, c2 ) ) return false;
  return true;
}

Candidate * TwoBodyCombiner::combine( const Candidate & c1, const Candidate & c2 ) {
  CompositeCandidate * cmp( new CompositeCandidate );
  cmp->addDaughter( c1 );
  cmp->addDaughter( c2 );
  cmp->set( addp4 );
  return cmp;
}

auto_ptr<TwoBodyCombiner::Candidates> 
TwoBodyCombiner::combine( const Candidates * src1, const Candidates * src2 ) {
  auto_ptr<Candidates> comps( new Candidates );
  if ( src1 == src2 ) {
    const Candidates & cands = * src1;
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
  } else {
    const Candidates & cands1 = * src1, & cands2 = * src2;
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
  }
  return comps;
}
