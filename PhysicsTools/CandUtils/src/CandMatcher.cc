#include "PhysicsTools/CandUtils/interface/CandMatcher.h"
#include <algorithm>
#include <iterator>
using namespace reco;
using namespace std;

CandMatcherBase::CandMatcherBase( const reco::CandMatchMap & map ) : 
  map_( map ) {
}

void CandMatcherBase::initMaps() {
  CandidateRefProd cands = map_.refProd().key;
  for( size_t i = 0; i < cands->size(); ++ i )
    candRefs_[ & (*cands)[ i ] ] = CandidateRef( cands, i );

  CandidateRefProd matched = map_.refProd().val; 
  for( size_t i = 0; i < matched->size(); ++ i )
    matchedRefs_[ & (*matched)[ i ] ] = CandidateRef( matched, i );

  matchedMothers_.resize( matched->size() );
  for( size_t i = 0; i < matched->size(); ++ i ) {
    const Candidate & c = (*matched)[ i ];
    for( Candidate::const_iterator d = c.begin(); d != c.end(); ++ d ) {
      vector<const Candidate *> daus = getDaughters( & * d );
      for( size_t j = 0; j < daus.size(); ++ j ) {
	const Candidate * daughter = daus[ j ];
	RefMap::const_iterator f = matchedRefs_.find( daughter );
	if ( f == matchedRefs_.end() ) continue;
	size_t k = f->second.key();
	assert( k < matchedMothers_.size() );
	matchedMothers_[ k ].push_back( i );
      }
    }
  }
}

CandMatcherBase::~CandMatcherBase() {
}

CandidateRef CandMatcherBase::operator()( const Candidate & c ) const {
  if ( c.hasMasterClone() ) {
    CandidateRef m = c.masterClone().castTo<CandidateRef>();
    if ( m.isNonnull() ) return (*this)( * m );
  }

  CandidateRefProd matched = map_.refProd().val; 
  unsigned int nDau = c.numberOfDaughters();
  if ( nDau > 0 ) {
    vector<size_t> moms;
    for( Candidate::const_iterator d = c.begin(); d != c.end(); ++ d ) {
      // check here generically if status == 3, then descend down to one more level
      CandidateRef m = (*this)( * d );
      if ( m.isNull() ) return CandidateRef();
      size_t mk = m.key();
      const vector<size_t> & allMomd = matchedMothers_[ mk ];
      vector<size_t> momd;
      for( size_t k = 0; k < allMomd.size(); ++ k ) {
	size_t m = allMomd[ k ];
	if( nDau == (*matched)[ m ].numberOfDaughters() )
	  momd.push_back( m );
      }
      if ( moms.size() == 0 ) moms = momd;
      else {
	vector<size_t> tmp;
	set_intersection( moms.begin(), moms.end(),
			  momd.begin(), momd.end(),
			  back_insert_iterator<vector<size_t> >( tmp ) );
	swap( moms, tmp );
      }
      if ( moms.size() == 0 ) return CandidateRef();
    }
    if ( moms.size() > 1 ) return CandidateRef();
    return CandidateRef( map_.refProd().val, moms.front() );
  }

  RefMap::const_iterator f = candRefs_.find( & c );
  if ( f != candRefs_.end() ) {
    CandidateRef ref = f->second;
    CandMatchMap::const_iterator f = map_.find( ref );
    if ( f != map_.end() ) {
      return f->val;
    }
  }
  
  return CandidateRef();
}

CandMatcher::CandMatcher( const CandMatchMap & map ) :
  CandMatcherBase( map ) {
  initMaps();
}

CandMatcher::~CandMatcher() {
}

vector<const Candidate *> CandMatcher::getDaughters( const Candidate * c ) const {
  vector<const Candidate *> v;
  v.push_back( c );
  return v;
}
