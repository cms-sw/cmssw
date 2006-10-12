#include "PhysicsTools/CandUtils/interface/ThreeBodyCombiner.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
using namespace reco;
using namespace std;

ThreeBodyCombiner::ThreeBodyCombiner( const parser::selector_ptr & sel, bool ck, const vector<int> & dauCharge, int q ) :
  checkCharge_( ck ), dauCharge_( dauCharge ), charge_( 0 ), overlap_(), select_( sel ) {
  if ( checkCharge_ ) charge_ = abs( q );
  assert( dauCharge_.size() == 3 );
}
	 
bool ThreeBodyCombiner::preselect( const Candidate & c0, const Candidate & c1,
				   const Candidate & c2 ) const {
  if ( checkCharge_ ) {
    if ( !
	 ( dauCharge_[0] == c0.charge() &&
	   dauCharge_[1] == c1.charge() &&
	   dauCharge_[2] == c2.charge() ) ||
	 ( dauCharge_[0] == - c0.charge() &&
	   dauCharge_[1] == - c1.charge() &&
	   dauCharge_[2] == - c2.charge() ) ) return false;
  }

  if ( overlap_( c0, c1 ) || overlap_( c0, c2 ) || overlap_( c1, c2 ) ) return false;

  return true;
}

Candidate * ThreeBodyCombiner::combine( const Candidate & c0, const Candidate & c1,
					const Candidate & c2 ) {
  CompositeCandidate * cmp( new CompositeCandidate );
  cmp->addDaughter( c0 );
  cmp->addDaughter( c1 );
  cmp->addDaughter( c2 );
  addp4_.set( * cmp );
  return cmp;
}

auto_ptr<CandidateCollection> 
ThreeBodyCombiner::combine( const CandidateCollection * src1, 
			    const CandidateCollection * src2,
			    const CandidateCollection * src3 ) {

  auto_ptr<CandidateCollection> comps( new CandidateCollection );

  if ( src1 == src2 && src1 == src3 ) {

    const CandidateCollection & cands = * src1;
    const int n = cands.size();
    for( int i1 = 0; i1 < n; ++ i1 ) {
      const Candidate & c1 = cands[ i1 ];
      for ( int i2 = i1 + 1; i2 < n; ++ i2 ) {
	const Candidate & c2 = cands[ i2 ];
	for( int i3 = i2 + 1; i3 < n; ++ i3 ) {
	  const Candidate & c3 = cands[ i3 ];
	  if ( preselect( c1, c2, c3 ) ) {
	    std::auto_ptr<Candidate> c( combine( c1, c2, c3 ) );
	    if ( select_( * c ) )
	      comps->push_back( c.release() );
	  }
	}
      }
    }
  } else if ( src1 == src2 ) {
    combineWithTwoEqualCollection( src1, src3, comps );
  } else if ( src1 == src3 ) {
    combineWithTwoEqualCollection( src1, src2, comps );
  } else if ( src2 == src3 ) {
    combineWithTwoEqualCollection( src2, src1, comps );
  } else {
    const CandidateCollection & cands1 = * src1, & cands2 = * src2,
      & cands3 = * src3;
    const int n1 = cands1.size(), n2 = cands2.size(), n3 = cands3.size();
    for( int i1 = 0; i1 < n1; ++ i1 ) {
      const Candidate & c1 = cands1[ i1 ];
      for ( int i2 = 0; i2 < n2; ++ i2 ) {
	const Candidate & c2 = cands2[ i2 ];
	for ( int i3 = 0; i3 < n3; ++ i3 ) {
	  const Candidate & c3 = cands3[ i3 ];
	  if ( preselect( c1, c2, c3 ) ) {
	    std::auto_ptr<Candidate> c( combine( c1, c2, c3 ) );
	    if ( select_( * c ) )
	      comps->push_back( c.release() );
	  }
	}
      }
    }
  }

  return comps;

}


void ThreeBodyCombiner::combineWithTwoEqualCollection( const CandidateCollection * equalSrc,
						       const CandidateCollection * diffSrc,
						       auto_ptr<CandidateCollection> comps ) {
  const CandidateCollection & cands = * equalSrc;
  const CandidateCollection & candsDiff = * diffSrc;
  const int n = cands.size();
  const int n3 = candsDiff.size();
  for( int i1 = 0; i1 < n; ++ i1 ) {
    const Candidate & c1 = cands[ i1 ];
    for ( int i2 = i1 + 1; i2 < n; ++ i2 ) {
      const Candidate & c2 = cands[ i2 ];
      for( int i3 = 0; i3 < n3; ++ i3 ) {
	const Candidate & c3 = candsDiff[ i3 ];
	if ( preselect( c1, c2, c3 ) ) {
	  std::auto_ptr<Candidate> c( combine( c1, c2, c3 ) );
	  if ( select_( * c ) )
	    comps->push_back( c.release() );
	}
      }
    }
  }
}
