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

template<typename C>
class CandMatcherBase {
public:
  /// map type
  typedef edm::AssociationMap<edm::OneToOne<C, reco::CandidateCollection> > map_type;
  /// map vector
  typedef std::vector<const map_type *> map_vector;
  /// concrete candidate reference type
  typedef typename map_type::key_type reference_type;
  /// concrete candidate reference type
  typedef typename reference_type::value_type value_type;
  /// constructor
  explicit CandMatcherBase( const map_vector & maps );
  /// destructor
  virtual ~CandMatcherBase();
  /// get match from transient reference
  reco::CandidateRef operator()( const reco::Candidate & ) const;

protected:
  /// get ultimate daughter (can skip status = 3 in MC)
  virtual std::vector<const reco::Candidate *> getDaughters( const reco::Candidate * ) const = 0;
  /// composite candidate preselection
  virtual bool compositePreselect( const reco::Candidate & c, const reco::Candidate & m ) const = 0;
  /// init maps
  void initMaps( const std::vector<const map_type *> & maps );

private:
  /// pointers to stored maps
  std::vector<const map_type *> maps_;
  /// reference to matched collectino
  reco::CandidateRefProd matched_;
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
  explicit CandMatcher( const typename CandMatcherBase<C>::map_vector & maps );
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

template<typename C>
CandMatcherBase<C>::CandMatcherBase( const typename CandMatcherBase<C>::map_vector & maps ):
  maps_( maps ) {
  reco::CandidateRefProd matched = maps.front()->refProd().val;
  for( typename map_vector::const_iterator m = maps.begin() + 1; 
       m != maps.end(); ++ m ) {
    if( (*m)->refProd().val != matched )
      throw edm::Exception( edm::errors::InvalidReference )
	<< "Multiple match maps specified matching different MC truth collections.\n"
	<< "Please, specify maps all matching to the same MC truth collection.\n"
	<< "In most of the cases you may want to match to genParticleCandidate.";
  }
}

template<typename C>
void CandMatcherBase<C>::initMaps( const std::vector<const typename CandMatcherBase<C>::map_type *> & maps ) {
  using namespace reco;
  using namespace std;
  for( typename map_vector::const_iterator m = maps.begin(); 
       m != maps.end(); ++ m ) {
    edm::RefProd<C> cands = (*m)->refProd().key;
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
  unsigned int nDau = c.numberOfDaughters();
  const CandidateCollection & matched = * matched_;
  if ( nDau > 0 ) {
    // check for composite candidate c
    // navigate to daughters and find parent matches
    vector<size_t> momsIntersection, momDaughters, tmp;
    for( Candidate::const_iterator d = c.begin(); d != c.end(); ++ d ) {
      // check here generically if status == 3, then descend down to one more level
      CandidateRef m = (*this)( * d );
      // if a daughter does not match, return a null ref.
      if ( m.isNull() ) return CandidateRef();
      // get matched mother indices (fetched previously)
      const vector<size_t> & allMomDaughters = matchedMothers_[ m.key() ];
      momDaughters.clear();
      for( vector<size_t>::const_iterator k = allMomDaughters.begin(); 
	   k != allMomDaughters.end(); ++ k ) {
	size_t m = * k;
	if( compositePreselect( c, matched[ m ] ) )
	  momDaughters.push_back( m );
      }
      // if no mother was found return null reference
      if ( momDaughters.size() == 0 ) return CandidateRef();
      // the first time, momsIntersection is set to momDaughters
      if ( momsIntersection.size() == 0 ) momsIntersection = momDaughters;
      else {
	tmp.clear();
	set_intersection( momsIntersection.begin(), momsIntersection.end(),
			  momDaughters.begin(), momDaughters.end(),
			  back_insert_iterator<vector<size_t> >( tmp ) );
	swap( momsIntersection, tmp );
      }
      if ( momsIntersection.size() == 0 ) return CandidateRef();
    }
    // if multiple mothers are found, return a null reference
    if ( momsIntersection.size() > 1 ) return CandidateRef();
    // return a reference to the unique mother
    return CandidateRef( matched_, momsIntersection.front() );
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
    return CandidateRef();
  }
}

template<typename C>
CandMatcher<C>::CandMatcher( const typename CandMatcherBase<C>::map_vector & maps ) :
  CandMatcherBase<C>( maps ) {
  CandMatcherBase<C>::initMaps( maps );
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

template<typename C>
bool CandMatcher<C>::compositePreselect( const reco::Candidate & c, const reco::Candidate & m ) const {
  // By default, check that the number of daughters is identical
  return( c.numberOfDaughters() == m.numberOfDaughters() );
}

#endif
