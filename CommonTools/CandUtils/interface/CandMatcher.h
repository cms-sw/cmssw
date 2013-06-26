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
#include "FWCore/Utilities/interface/EDMException.h"
#include "CommonTools/CandUtils/interface/CandMapTrait.h"

template<typename C1, typename C2 = C1>
class CandMatcherBase {
public:
  /// map type
  typedef typename reco::helper::CandMapTrait<C1, C2>::type map_type;
  /// ref type
  typedef typename reco::helper::CandRefTrait<C2>::ref_type ref_type;
  /// refProd type
  typedef typename reco::helper::CandRefTrait<C2>::refProd_type refProd_type;
  /// map vector
  typedef std::vector<const map_type *> map_vector;
  /// concrete candidate reference type
  typedef typename map_type::key_type reference_type;
  /// concrete candidate reference type
  typedef typename reference_type::value_type value_type;
  /// constructor
  explicit CandMatcherBase( const map_vector & maps );
  /// constructor
  explicit CandMatcherBase( const map_type & map );
 /// destructor
  virtual ~CandMatcherBase();
  /// get match from transient reference
  ref_type operator()( const reco::Candidate & ) const;

protected:
  /// get ultimate daughter (can skip status = 3 in MC)
  virtual std::vector<const reco::Candidate *> getDaughters( const reco::Candidate * ) const = 0;
  /// composite candidate preselection
  virtual bool compositePreselect( const reco::Candidate & c, const reco::Candidate & m ) const = 0;
  /// init maps
  void initMaps();

protected:
  const std::vector<const map_type *> & maps() const { return maps_; }
private:
  /// pointers to stored maps
  std::vector<const map_type *> maps_;
  /// reference to matched collectino
  refProd_type matched_;
  /// pointer map type
  typedef std::map<const reco::Candidate *, reference_type> CandRefMap;
  /// pointer map type
  typedef std::map<const reco::Candidate *, ref_type> MatchedRefMap;
  /// pointer map of candidates (e.g.: reco)
  CandRefMap candRefs_;
  /// pointer map of matched candidates (e.g.: MC truth)
  MatchedRefMap matchedRefs_;
  /// mother + n.daughters indices from matched
  std::vector<std::set<size_t> > matchedMothers_;
  /// init at constructor
  void init();
};

template<typename C1, typename C2 = C1>
class CandMatcher : public CandMatcherBase<C1, C2> {
public:
  /// constructor
  explicit CandMatcher( const typename CandMatcherBase<C1, C2>::map_vector & maps );
  /// constructor
  explicit CandMatcher( const typename CandMatcherBase<C1, C2>::map_type & map );
  /// destructor
  virtual ~CandMatcher();

protected:
  /// get ultimate daughter (get all in the general case)
  virtual std::vector<const reco::Candidate *> getDaughters( const reco::Candidate * ) const;
  /// composite candidate preselection
  virtual bool compositePreselect( const reco::Candidate & c, const reco::Candidate & m ) const;
};

#include <algorithm>
#include <iterator>

template<typename C1, typename C2>
void CandMatcherBase<C1, C2>::init() {
  matched_ = maps_.front()->refProd().val;
  for( typename map_vector::const_iterator m = maps_.begin() + 1; 
       m != maps_.end(); ++ m ) {
    if( (*m)->refProd().val != matched_ )
      throw edm::Exception( edm::errors::InvalidReference )
	<< "Multiple match maps specified matching different MC truth collections.\n"
	<< "Please, specify maps all matching to the same MC truth collection.\n"
	<< "In most of the cases you may want to match to genParticleCandidate.";
  }
}

template<typename C1, typename C2>
CandMatcherBase<C1, C2>::CandMatcherBase( const typename CandMatcherBase<C1, C2>::map_vector & maps ):
  maps_( maps ) {
  init();
}

template<typename C1, typename C2>
CandMatcherBase<C1, C2>::CandMatcherBase( const typename CandMatcherBase<C1, C2>::map_type & map ):
  maps_( 1, & map ) {
  init();
}

template<typename C1, typename C2>
void CandMatcherBase<C1, C2>::initMaps() {
  using namespace reco;
  using namespace std;
  for( typename map_vector::const_iterator m = maps_.begin(); 
       m != maps_.end(); ++ m ) {
    typename CandMatcherBase<C1, C2>::map_type::ref_type::key_type cands = (*m)->refProd().key;
    for( size_t i = 0; i < cands->size(); ++ i ) {
      candRefs_[ & (*cands)[ i ] ] = reference_type( cands, i );
    } 
    const C2 & matched = * matched_;
    size_t matchedSize = matched.size();
    for( size_t i = 0; i < matchedSize; ++ i )
      matchedRefs_[ & matched[ i ] ] = ref_type( matched_, i );
    matchedMothers_.resize( matchedSize );
    for( size_t i = 0; i < matchedSize; ++ i ) {
      const Candidate & c = matched[ i ];
      for( Candidate::const_iterator d = c.begin(); d != c.end(); ++ d ) {
	vector<const Candidate *> daus = getDaughters( & * d );
	for( size_t j = 0; j < daus.size(); ++ j ) {
	  const Candidate * daughter = daus[ j ];
	  typename MatchedRefMap::const_iterator f = matchedRefs_.find( daughter );
	  if ( f == matchedRefs_.end() ) continue;
	  size_t k = f->second.key();
	  assert( k < matchedMothers_.size() );
	  matchedMothers_[ k ].insert( i );
	}
      }
    }
  }
}

template<typename C1, typename C2>
CandMatcherBase<C1, C2>::~CandMatcherBase() {
}

template<typename C1, typename C2>
typename CandMatcherBase<C1, C2>::ref_type CandMatcherBase<C1, C2>::operator()( const reco::Candidate & c ) const {
  using namespace reco;
  using namespace std;
  if ( c.hasMasterClone() )
    return (*this)( * c.masterClone() );
  unsigned int nDau = c.numberOfDaughters();
  const C2 & matched = * matched_;
  if ( nDau > 0 ) {
    // check for composite candidate c
    // navigate to daughters and find parent matches
    set<size_t> momsIntersection, momDaughters, tmp;
    for( Candidate::const_iterator d = c.begin(); d != c.end(); ++ d ) {
      // check here generically if status == 3, then descend down to one more level
      ref_type m = (*this)( * d );
      // if a daughter does not match, return a null ref.
      if ( m.isNull() ) return ref_type();
      // get matched mother indices (fetched previously)
      const set<size_t> & allMomDaughters = matchedMothers_[ m.key() ];
      momDaughters.clear();
      for( set<size_t>::const_iterator k = allMomDaughters.begin(); 
	   k != allMomDaughters.end(); ++ k ) {
	size_t m = * k;
	if( compositePreselect( c, matched[ m ] ) )
	  momDaughters.insert( m );
      }
      // if no mother was found return null reference
      if ( momDaughters.size() == 0 ) return ref_type();
      // the first time, momsIntersection is set to momDaughters
      if ( momsIntersection.size() == 0 ) momsIntersection = momDaughters;
      else {
	tmp.clear();
	set_intersection( momsIntersection.begin(), momsIntersection.end(),
			  momDaughters.begin(), momDaughters.end(),
			 inserter( tmp, tmp.begin() ) );
	swap( momsIntersection, tmp );
      }
      if ( momsIntersection.size() == 0 ) return ref_type();
    }
    // if multiple mothers are found, return a null reference
    if ( momsIntersection.size() > 1 ) return ref_type();
    // return a reference to the unique mother
    return ref_type( matched_, * momsIntersection.begin() );
  } else {
    // check for non-composite (leaf) candidate 
    // if one of the maps contains the candidate c
    for( typename std::vector<const map_type *>::const_iterator m = maps_.begin(); 
	 m != maps_.end(); ++ m ) {
      typename CandRefMap::const_iterator f = candRefs_.find( & c );
      if ( f != candRefs_.end() ) {
	reference_type ref = f->second;
	typename map_type::const_iterator f = (*m)->find( ref );
	if ( f != (*m)->end() ) {
	  return f->val;
	}
      }
    }
    return ref_type();
  }
}

template<typename C1, typename C2>
CandMatcher<C1, C2>::CandMatcher( const typename CandMatcherBase<C1, C2>::map_vector & maps ) :
  CandMatcherBase<C1, C2>( maps ) {
  CandMatcherBase<C1, C2>::initMaps();
}

template<typename C1, typename C2>
CandMatcher<C1, C2>::CandMatcher( const typename CandMatcherBase<C1, C2>::map_type & map ) :
  CandMatcherBase<C1, C2>( map ) {
  CandMatcherBase<C1, C2>::initMaps();
}

template<typename C1, typename C2>
CandMatcher<C1, C2>::~CandMatcher() {
}

template<typename C1, typename C2>
std::vector<const reco::Candidate *> CandMatcher<C1, C2>::getDaughters( const reco::Candidate * c ) const {
  std::vector<const reco::Candidate *> v;
  v.push_back( c );
  return v;
}

template<typename C1, typename C2>
bool CandMatcher<C1, C2>::compositePreselect( const reco::Candidate & c, const reco::Candidate & m ) const {
  // By default, check that the number of daughters is identical
  return( c.numberOfDaughters() == m.numberOfDaughters() );
}

#endif
