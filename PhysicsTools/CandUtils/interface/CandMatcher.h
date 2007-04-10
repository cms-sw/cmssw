#ifndef CandUtils_CandMatcher_h
#define CandUtils_CandMatcher_h
/* class CandMatcher
 *
 * \author Luca Lista, INFN
 *
 */
#include <set>
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToOne.h"

template<typename C>
class CandMatcherBase {
public:
  /// map type
  typedef edm::AssociationMap<edm::OneToOne<C, reco::CandidateCollection> > map_type;
  /// concrete candidate reference type
  typedef typename map_type::key_type reference_type;
  /// concrete candidate reference type
  typedef typename reference_type::value_type value_type;
  /// constructor
  explicit CandMatcherBase( const map_type & map );
  /// destructor
  virtual ~CandMatcherBase();
  /// get match from transient reference
  reco::CandidateRef operator()( const reco::Candidate & ) const;

protected:
  /// get ultimate daughter (can skip status = 3 in MC)
  virtual std::vector<const reco::Candidate *> getDaughters( const reco::Candidate * ) const = 0;
  /// init maps
  void initMaps();

private:
  /// reference to match map, typically taken from the event
  const map_type & map_;
  /// reference to matched collectino
  const reco::CandidateRefProd matched_;
  /// pointer map type
  typedef std::map<const reco::Candidate *, reference_type> CandRefMap;
  /// pointer map type
  typedef std::map<const reco::Candidate *, reco::CandidateRef> MatchedRefMap;
  /// pointer map of candidates (e.g.: reco)
  CandRefMap candRefs_;
  /// pointer map of matched candidates (e.g.: MC truth)
  MatchedRefMap matchedRefs_;
  /// mother + n.daughters indices from matched
  std::vector<std::vector<size_t> > matchedMothers_;
};

template<typename C>
class CandMatcher : public CandMatcherBase<C> {
public:
  /// constructor
  explicit CandMatcher( const typename CandMatcherBase<C>::map_type & map );
  /// destructor
  virtual ~CandMatcher();

protected:
  /// get ultimate daughter (get all in the general case)
  virtual std::vector<const reco::Candidate *> getDaughters( const reco::Candidate * ) const;
};

#include <algorithm>
#include <iterator>

template<typename C>
CandMatcherBase<C>::CandMatcherBase( const typename CandMatcherBase<C>::map_type & map ) : 
  map_( map ),
  matched_( map_.refProd().val ) {
}

template<typename C>
void CandMatcherBase<C>::initMaps() {
  using namespace reco;
  using namespace std;
  edm::RefProd<C> cands = map_.refProd().key;
  for( size_t i = 0; i < cands->size(); ++ i )
    candRefs_[ & (*cands)[ i ] ] = reference_type( cands, i );
  const CandidateCollection & matched = * matched_;
  for( size_t i = 0; i < matched.size(); ++ i )
    matchedRefs_[ & matched[ i ] ] = CandidateRef( matched_, i );
  matchedMothers_.resize( matched.size() );
  for( size_t i = 0; i < matched.size(); ++ i ) {
    const Candidate & c = matched[ i ];
    for( Candidate::const_iterator d = c.begin(); d != c.end(); ++ d ) {
      vector<const Candidate *> daus = getDaughters( & * d );
      for( size_t j = 0; j < daus.size(); ++ j ) {
	const Candidate * daughter = daus[ j ];
	typename MatchedRefMap::const_iterator f = matchedRefs_.find( daughter );
	if ( f == matchedRefs_.end() ) continue;
	size_t k = f->second.key();
	assert( k < matchedMothers_.size() );
	matchedMothers_[ k ].push_back( i );
      }
    }
  }
}

template<typename C>
CandMatcherBase<C>::~CandMatcherBase() {
}

template<typename C>
reco::CandidateRef CandMatcherBase<C>::operator()( const reco::Candidate & c ) const {
  using namespace reco;
  using namespace std;
  if ( c.hasMasterClone() )
    return (*this)( * c.masterClone() );
  const CandidateCollection & matched = * matched_;
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
	if( nDau <= matched[ m ].numberOfDaughters() )
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
    return CandidateRef( matched_, moms.front() );
  }

  typename CandRefMap::const_iterator f = candRefs_.find( & c );
  if ( f != candRefs_.end() ) {
    reference_type ref = f->second;
    typename map_type::const_iterator f = map_.find( ref );
    if ( f != map_.end() ) {
      return f->val;
    }
  }
  
  return CandidateRef();
}

template<typename C>
CandMatcher<C>::CandMatcher( const typename CandMatcherBase<C>::map_type & map ) :
  CandMatcherBase<C>( map ) {
  CandMatcherBase<C>::initMaps();
}

template<typename C>
CandMatcher<C>::~CandMatcher() {
}

template<typename C>
std::vector<const reco::Candidate *> CandMatcher<C>::getDaughters( const reco::Candidate * c ) const {
  std::vector<const reco::Candidate *> v;
  v.push_back( c );
  return v;
}

#endif
