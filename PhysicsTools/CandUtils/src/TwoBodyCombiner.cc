// $Id: TwoBodyCombiner.cc,v 1.13 2006/07/26 08:48:06 llista Exp $
#include "PhysicsTools/CandUtils/interface/TwoBodyCombiner.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
using namespace reco;
using namespace std;

TwoBodyCombiner::TwoBodyCombiner( const parser::selector_ptr & sel, bool ck, int q ) :
  checkCharge_( ck ), charge_( 0 ), overlap_(), select_( sel ) {
  if ( checkCharge_ ) charge_ = abs( q );
}
	 
bool TwoBodyCombiner::preselect( const Candidate & c1, const Candidate & c2 ) const {
  if ( checkCharge_ ) {
    if ( charge_ != abs( c1.charge() + c2.charge() ) ) return false;
  }
  if ( overlap_( c1, c2 ) ) return false;
  return true;
}

Candidate * TwoBodyCombiner::combine( const Candidate & c1, const Candidate & c2 ) {
  CompositeCandidate * cmp( new CompositeCandidate );
  cmp->addDaughter( c1 );
  cmp->addDaughter( c2 );
  addp4_.set( * cmp );
  return cmp;
}

auto_ptr<CandidateCollection> 
TwoBodyCombiner::combine( const CandidateCollection * src1, const CandidateCollection * src2 ) {
  auto_ptr<CandidateCollection> comps( new CandidateCollection );
  if ( src1 == src2 ) {
    const CandidateCollection & cands = * src1;
    const int n = cands.size();
    for( int i1 = 0; i1 < n; ++ i1 ) {
      const Candidate & c1 = cands[ i1 ];
      for ( int i2 = i1 + 1; i2 < n; ++ i2 ) {
	const Candidate & c2 = cands[ i2 ];
	if ( preselect( c1, c2 ) ) {
	  std::auto_ptr<Candidate> c( combine( c1, c2 ) );
	  if ( select_( * c ) )
	    comps->push_back( c.release() );
	}
      }
    }
  } else {
    const CandidateCollection & cands1 = * src1, & cands2 = * src2;
    const int n1 = cands1.size(), n2 = cands2.size();
    for( int i1 = 0; i1 < n1; ++ i1 ) {
      const Candidate & c1 = cands1[ i1 ];
      for ( int i2 = 0; i2 < n2; ++ i2 ) {
	const Candidate & c2 = cands2[ i2 ];
	if ( preselect( c1, c2 ) ) {
	  std::auto_ptr<Candidate> c( combine( c1, c2 ) );
	  if ( select_( * c ) )
	    comps->push_back( c.release() );
	}
      }
    }
  }
  return comps;
}
