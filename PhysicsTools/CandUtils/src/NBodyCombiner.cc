#include "PhysicsTools/CandUtils/interface/NBodyCombiner.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
using namespace reco;
using namespace std;

NBodyCombiner::NBodyCombiner( const parser::SelectorPtr & sel, bool ck, const vector<int> & dauCharge ) :
  checkCharge_( ck ), dauCharge_( dauCharge ), charge_( 0 ), overlap_(), select_( sel ) {
}

bool NBodyCombiner::preselect( const Candidate & c1, const Candidate & c2 ) const {
  if ( checkCharge_ ) {
    if ( charge_ != abs( c1.charge() + c2.charge() ) ) return false;
  }
  if ( overlap_( c1, c2 ) ) return false;
  return true;
}

Candidate * NBodyCombiner::combine( const Candidate & c1, const Candidate & c2 ) const {
  CompositeCandidate * cmp( new CompositeCandidate );
  cmp->addDaughter( c1 );
  cmp->addDaughter( c2 );
  addp4_.set( * cmp );
  return cmp;
}

auto_ptr<CandidateCollection> 
NBodyCombiner::combine( const vector<const CandidateCollection * > & src ) const {
  auto_ptr<CandidateCollection> comps( new CandidateCollection );
  
  if( src.size() == 2 ) {
    const CandidateCollection * src1 = src[ 0 ], * src2 = src[ 1 ];
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
  } else {
    const CandidateCollection & src0 = * src[ 0 ];
    for( CandidateCollection::const_iterator  c = src0.begin(); c != src0.end(); ++ c ) {
      bool sameCharge = ( c->charge() == dauCharge_[ 0 ] );
      vector<const Candidate *> cv;
      cv.push_back( & * c );
      combine( 1, sameCharge, cv, src.begin() + 1, src.end(), comps );
    }
  }

  return comps;
}


void NBodyCombiner::combine( size_t collectionIndex, bool sameCharge, vector<const Candidate *> cv,
			     const vector<const CandidateCollection * >::const_iterator begin,
			     const vector<const CandidateCollection * >::const_iterator end,
			     auto_ptr<CandidateCollection> & comps
			     ) const {
  if( begin == end ) {
    CompositeCandidate * cmp( new CompositeCandidate );
    for( vector<const Candidate*>::const_iterator i = cv.begin(); i != cv.end(); ++ i )
      cmp->addDaughter( * * i );
    addp4_.set( * cmp );
    comps->push_back( cmp );
  } else {
    const CandidateCollection & src = * * begin;
    for( CandidateCollection::const_iterator  c = src.begin(); c != src.end(); ++ c ) {
      bool same = ( c->charge() == dauCharge_[ collectionIndex ] );
      if ( sameCharge == same ) {
	for( vector<const Candidate *>::const_iterator i = cv.begin(); i != cv.end(); ++i )
	  if ( overlap_( * c, ** i ) ) return;
	cv.push_back( & * c );
	combine( collectionIndex + 1, sameCharge, cv, begin + 1, end, comps );
      }
    }
  }
}

