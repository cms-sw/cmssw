#include "PhysicsTools/CandUtils/interface/CandMatcher.h"
#include <algorithm>
#include <iterator>
using namespace reco;
using namespace std;

CandMatcher::CandMatcher( const reco::CandMatchMap & map ) : 
  map_( map ) {
  /// store pointer map with candidates (e.g.: reco)
  CandidateRefProd cands = map_.refProd().key;
  for( size_t i = 0; i < cands->size(); ++ i )
    candRefs_[ & (*cands)[ i ] ] = CandidateRef( cands, i );

  /// store pointer map with matched candidates (e.g. MC truth)
  CandidateRefProd matched = map_.refProd().val; 
  for( size_t i = 0; i < matched->size(); ++ i )
    matchedRefs_[ & (*matched)[ i ] ] = CandidateRef( matched, i );

  /// fill mother indices looping over daughters.
  /// there could be more than one mother per daughter
  mothers_.resize( matched->size() );
  for( size_t i = 0; i < matched->size(); ++ i ) {
    const Candidate & c = (*matched)[ i ];
    for( Candidate::const_iterator d = c.begin(); d != c.end(); ++ d ) {
      RefMap::const_iterator f = matchedRefs_.find( & * d );
      if ( f == matchedRefs_.end() ) continue;
      CandidateRef r = f->second;
      mothers_[ r.key() ].push_back( i );
    }
  }
}

CandidateRef CandMatcher::operator()( const CandidateRef & c ) const {
  /// find candidate in match map
  CandMatchMap::const_iterator f = map_.find( c );
  /// if found (i.e.: it's a "leaf" daughter), return matched candidate
  if ( f != map_.end() ) return f->val;

  /// recursively find matched daughters
  vector<size_t> moms;
  for( Candidate::const_iterator d = c->begin(); d != c->end(); ++ d ) {
    /// find matched daughter
    CandidateRef m = (*this)( * d );
    /// if not matched, return void reference
    if ( m.isNull() ) return CandidateRef();
    /// get mother indices
    const vector<size_t> & momd = mothers_[ m.key() ];
    /// if empty (it's the first iteration) get first 
    /// daughter's associated mother indices
    if ( moms.size() == 0 ) moms = momd;
    else {
      /// intersect mother indices sets
      vector<size_t> tmp;
      set_intersection( moms.begin(), moms.end(),
			momd.begin(), momd.end(),
			back_insert_iterator<vector<size_t> >( tmp ) );
      swap( moms, tmp );
    }
    /// if the intersection is empty (no common mother 
    /// indices) return void reference
    if ( moms.size() == 0 ) return CandidateRef();
  }
  /// return void reference if more than one matched mother
  if ( moms.size() > 1 ) return CandidateRef();
  /// return matched mother
  return CandidateRef( map_.refProd().val, moms.front() );
}

CandidateRef CandMatcher::operator()( const Candidate & c ) const {
  /// find a perssitent reference in pointer map
  RefMap::const_iterator f = candRefs_.find( & c );
  /// if found, return match from persistent reference
  if ( f != candRefs_.end() ) return (*this)( f->second );
  /// try to find a master clone
  if ( c.hasMasterClone() ) {
    CandidateRef m = c.masterClone().castTo<CandidateRef>();
    if ( m.isNull() ) return CandidateRef();
    else return (*this)( m );
  } else {
    return CandidateRef();
  }
}
