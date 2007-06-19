#include "PhysicsTools/CandUtils/interface/CandCombiner.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "FWCore/Utilities/interface/EDMException.h"

using namespace reco;
using namespace std;

CandCombinerBase::CandCombinerBase() :
  checkCharge_( false ), dauCharge_(), overlap_() {
}

CandCombinerBase::CandCombinerBase( int q1, int q2 ) :
  checkCharge_( true ), dauCharge_( 2 ), overlap_() {
  dauCharge_[ 0 ] = q1;
  dauCharge_[ 1 ] = q2;
}

CandCombinerBase::CandCombinerBase( int q1, int q2, int q3 ) :
  checkCharge_( true ), dauCharge_( 3 ), overlap_() {
  dauCharge_[ 0 ] = q1;
  dauCharge_[ 1 ] = q2;
  dauCharge_[ 2 ] = q3;
}

CandCombinerBase::CandCombinerBase( int q1, int q2, int q3, int q4 ) :
  checkCharge_( true ), dauCharge_( 4 ), overlap_() {
  dauCharge_[ 0 ] = q1;
  dauCharge_[ 1 ] = q2;
  dauCharge_[ 2 ] = q3;
  dauCharge_[ 3 ] = q4;
}

CandCombinerBase::CandCombinerBase( bool checkCharge, const vector<int> & dauCharge ) :
  checkCharge_( checkCharge ), dauCharge_( dauCharge ), overlap_() {
}

CandCombinerBase::~CandCombinerBase() {
}

bool CandCombinerBase::preselect( const Candidate & c1, const Candidate & c2 ) const {
  if ( checkCharge_ ) {
    int dq1 = dauCharge_[0], dq2 = dauCharge_[1], q1 = c1.charge(), q2 = c2.charge();
    bool matchCharge = ( q1 == dq1 && q2 == dq2 ) || ( q1 == -dq1 && q2 == -dq2 ); 
    if (!matchCharge) return false; 
  }
  if ( overlap_( c1, c2 ) ) return false;
  return selectPair( c1, c2 );
}

reco::Candidate * CandCombinerBase::combine( const reco::CandidateRef & c1, const reco::CandidateRef & c2 ) const {
  CompositeCandidate * cmp( new CompositeCandidate );
  addDaughter( cmp, c1 );
  addDaughter( cmp, c2 );
  setup( cmp );
  return cmp;
}

auto_ptr<CandidateCollection> 
CandCombinerBase::combine( const vector<CandidateRefProd> & src ) const {
  size_t srcSize = src.size();
  if ( checkCharge_ && dauCharge_.size() != srcSize )
    throw edm::Exception( edm::errors::Configuration ) 
      << "CandCombiner: trying to combine " << srcSize << " collections"
      << " but configured to check against " << dauCharge_.size() << " charges"
      << endl;
  
  auto_ptr<CandidateCollection> comps( new CandidateCollection );
  
  if( srcSize == 2 ) {
    CandidateRefProd src1 = src[ 0 ], src2 = src[ 1 ];
    if ( src1 == src2 ) {
      const CandidateCollection & cands = * src1;
      const int n = cands.size();
      for( int i1 = 0; i1 < n; ++ i1 ) {
	const Candidate & c1 = cands[ i1 ];
	CandidateRef cr1( src1, i1 );
	for ( int i2 = i1 + 1; i2 < n; ++ i2 ) {
	  const Candidate & c2 = cands[ i2 ];
	  if ( preselect( c1, c2 ) ) {
	    CandidateRef cr2( src2, i2 );
	    std::auto_ptr<Candidate> c( combine( cr1, cr2 ) );
	    if ( select( * c ) )
	      comps->push_back( c.release() );
	  }
	}
      }
    } else {
      const CandidateCollection & cands1 = * src1, & cands2 = * src2;
      const int n1 = cands1.size(), n2 = cands2.size();
      for( int i1 = 0; i1 < n1; ++ i1 ) {
	const Candidate & c1 = cands1[ i1 ];
	CandidateRef cr1( src1, i1 );
	for ( int i2 = 0; i2 < n2; ++ i2 ) {
	  const Candidate & c2 = cands2[ i2 ];
	  if ( preselect( c1, c2 ) ) {
	    CandidateRef cr2( src2, i2 );
	    std::auto_ptr<Candidate> c( combine( cr1, cr2 ) );
	    if ( select( * c ) )
	      comps->push_back( c.release() );
	  }
	}
      }
    }
  } else {
    CandStack stack;
    ChargeStack qStack;
    combine( 0, stack, qStack, src.begin(), src.end(), comps );
  }

  return comps;
}

auto_ptr<CandidateCollection> 
CandCombinerBase::combine( const CandidateRefProd & src ) const {
  if ( checkCharge_ && dauCharge_.size() != 2 )
    throw edm::Exception( edm::errors::Configuration ) 
      << "CandCombiner: trying to combine 2 collections"
      << " but configured to check against " << dauCharge_.size() << " charges"
      << endl;

  auto_ptr<CandidateCollection> comps( new CandidateCollection );
  const CandidateCollection & cands = * src; 
  const int n = cands.size();
  for( int i1 = 0; i1 < n; ++ i1 ) {
    const Candidate & c1 = cands[ i1 ];
    CandidateRef cr1( src, i1 );
    for ( int i2 = i1 + 1; i2 < n; ++ i2 ) {
      const Candidate & c2 = cands[ i2 ];
      if ( preselect( c1, c2 ) ) {
	CandidateRef cr2( src, i2 );
	std::auto_ptr<Candidate> c( combine( cr1, cr2 ) );
	if ( select( * c ) )
	  comps->push_back( c.release() );
      }
    } 
  }

  return comps;
}

auto_ptr<CandidateCollection> 
CandCombinerBase::combine( const CandidateRefProd & src1, const CandidateRefProd & src2 ) const {
  vector<CandidateRefProd> src;
  src.push_back( src1 );
  src.push_back( src2 );
  return combine( src );
}

auto_ptr<CandidateCollection> 
CandCombinerBase::combine( const CandidateRefProd & src1, const CandidateRefProd & src2, const CandidateRefProd & src3 ) const {
  vector<CandidateRefProd> src;
  src.push_back( src1 );
  src.push_back( src2 );
  src.push_back( src3 );
  return combine( src );
}

auto_ptr<CandidateCollection> 
CandCombinerBase::combine( const CandidateRefProd & src1, const CandidateRefProd & src2, 
			   const CandidateRefProd & src3, const CandidateRefProd & src4 ) const {
  vector<CandidateRefProd> src;
  src.push_back( src1 );
  src.push_back( src2 );
  src.push_back( src3 );
  src.push_back( src4 );
  return combine( src );
}

void CandCombinerBase::combine( size_t collectionIndex, CandStack & stack, ChargeStack & qStack,
				 vector<CandidateRefProd>::const_iterator collBegin,
				 vector<CandidateRefProd>::const_iterator collEnd,
				 auto_ptr<CandidateCollection> & comps
				 ) const {
  if( collBegin == collEnd ) {
    static const int undetermined = 0, sameDecay = 1, conjDecay = -1, wrongDecay = 2;
    int decayType = undetermined;
    if (checkCharge_) {
      assert( qStack.size() == stack.size() );
      for( size_t i = 0; i < qStack.size(); ++i ) {
	int q = qStack[i], dq = dauCharge_[i];
	if ( decayType == undetermined ) {
	  if ( q != 0 && dq != 0 ) {
	    if ( q == dq ) decayType = sameDecay;
	    else if ( q == -dq ) decayType = conjDecay;
	    else decayType = wrongDecay;
	  }
	} else if ( ( decayType == sameDecay && q != dq ) ||
		    ( decayType == conjDecay && q != -dq ) ) {
	  decayType = wrongDecay;
	}
	if ( decayType == wrongDecay ) break;
      }
    }
    if ( decayType != wrongDecay ) { 
      CompositeCandidate * cmp( new CompositeCandidate );
      for( CandStack::const_iterator i = stack.begin(); i != stack.end(); ++ i ) {
	addDaughter( cmp, i->first );
      }
      setup( cmp );
      if ( select( * cmp ) )
	comps->push_back( cmp );
    }
  } else {
    const CandidateRefProd srcRef = * collBegin;
    const CandidateCollection & src = * srcRef;
    size_t candBegin = 0, candEnd = src.size();
    for( CandStack::const_iterator i = stack.begin(); i != stack.end(); ++i ) 
      if ( * collBegin == * i->second ) 
	candBegin = i->first.key() + 1;
    for( size_t candIndex = candBegin; candIndex != candEnd; ++ candIndex ) {
      CandidateRef cand( srcRef, candIndex );
      bool noOverlap = true;
      for( CandStack::const_iterator i = stack.begin(); i != stack.end(); ++i ) 
	if ( overlap_( * cand, * ( i->first ) ) ) { 
	  noOverlap = false; 
	  break; 
	}
      if ( noOverlap ) {
	stack.push_back( make_pair( cand, collBegin ) );
	if ( checkCharge_ ) qStack.push_back( cand->charge() ); 
	combine( collectionIndex + 1, stack, qStack, collBegin + 1, collEnd, comps );
	stack.pop_back();
	qStack.pop_back();
      }
    }
  }
}



