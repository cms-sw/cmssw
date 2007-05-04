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
    vector<size_t> moms;
    for( Candidate::const_iterator d = c.begin(); d != c.end(); ++ d ) {
      // check here generically if status == 3, then descend down to one more level
      CandidateRef mom = (*this)( * d );
      if ( mom.isNull() ) return CandidateRef();
      size_t mk = mom.key();
      const vector<size_t> & allMomd = matchedMothers_[ mk ];
      vector<size_t> momd;
      for( size_t k = 0; k < allMomd.size(); ++ k ) {
	size_t mom = allMomd[ k ];
	if( nDau <= matched[ mom ].numberOfDaughters() )
	  momd.push_back( mom );
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

#endif
