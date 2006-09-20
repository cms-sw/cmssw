#include "PhysicsTools/CandUtils/interface/NBodyCombiner.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
using namespace reco;
using namespace std;

NBodyCombiner::NBodyCombiner( const parser::SelectorPtr & sel, bool ck, const vector<int> & dauCharge ) :
  checkCharge_( ck ), dauCharge_( dauCharge ), overlap_(), select_( sel ) {
}

NBodyCombiner::ChargeInfo NBodyCombiner::chargeInfo( int q1, int q2 ) {
  if ( q1 == q2 ) return same;
  if ( q1 == - q2 ) return opposite;
  return invalid;
}


bool NBodyCombiner::preselect( const Candidate & c1, const Candidate & c2 ) const {
  if ( checkCharge_ ) {
    ChargeInfo ch1 = chargeInfo( c1.charge(), dauCharge_[ 0 ] );
    if ( ch1 == invalid ) return false;
    ChargeInfo ch2 = chargeInfo( c2.charge(), dauCharge_[ 1 ] );
    if ( ch2 == invalid ) return false;
    if ( ch1 != ch2 ) return false;
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
    CandStack stack;
    combine( 0, undetermined, stack, src.begin(), src.end(), comps );
  }

  return comps;
}


void NBodyCombiner::combine( size_t collectionIndex, ChargeInfo chkCharge, CandStack & stack,
			     vector<const CandidateCollection * >::const_iterator collBegin,
			     vector<const CandidateCollection * >::const_iterator collEnd,
			     auto_ptr<CandidateCollection> & comps
			     ) const {
  if( collBegin == collEnd ) {
    CompositeCandidate * cmp( new CompositeCandidate );
    for( CandStack::const_iterator i = stack.begin(); i != stack.end(); ++ i )
      cmp->addDaughter( * ( i->first ) );
    addp4_.set( * cmp );
    if ( select_( * cmp ) )
      comps->push_back( cmp );
  } else {
    const CandidateCollection & src = * * collBegin;
    CandidateCollection::const_iterator candBegin = src.begin(), candEnd = src.end();
    for( CandStack::const_iterator i = stack.begin(); i != stack.end(); ++i ) 
      if ( * collBegin == * i->second ) 
	candBegin = i->first + 1;
    for( CandidateCollection::const_iterator  cand = candBegin; cand != candEnd; ++ cand ) {
      if ( checkCharge_ ) {
	int q = cand->charge();
	ChargeInfo ch = chargeInfo( q, dauCharge_[ collectionIndex ] );
	if( ch == invalid ) continue;
	if ( chkCharge == undetermined && q != 0 ) chkCharge = ch;
      }
      bool noOverlap = true;
      for( CandStack::const_iterator i = stack.begin(); i != stack.end(); ++i ) 
	if ( overlap_( * cand, * ( i->first ) ) ) { 
	  noOverlap = false; 
	  break; 
	}
      if ( noOverlap ) {
	stack.push_back( make_pair( cand, collBegin ) );
	combine( collectionIndex + 1, chkCharge, stack, collBegin + 1, collEnd, comps );
	stack.pop_back();
      }
    }
  }
}

