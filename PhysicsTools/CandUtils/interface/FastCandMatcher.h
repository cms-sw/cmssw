#ifndef CandUtils_FastCandMatcher_h
#define CandUtils_FastCandMatcher_h
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
class FastCandMatcher {
public:
  /// map type
  typedef edm::AssociationMap<edm::OneToOne<C, reco::CandidateCollection> > map_type;
  /// map vector
  typedef std::vector<const map_type *> map_vector;
  /// constructor
  explicit FastCandMatcher( const map_vector & maps );
  /// constructor
  explicit FastCandMatcher( const map_type & map );
  /// get match from transient reference
  const reco::Candidate * operator()( const reco::Candidate & ) const;

protected:
  const std::vector<const map_type *> & maps() const { return maps_; }
private:
  /// pointers to stored maps
  std::vector<const map_type *> maps_;
};

template<typename C>
FastCandMatcher<C>::FastCandMatcher( const typename FastCandMatcher<C>::map_vector & maps ):
  maps_( maps ) {
}

template<typename C>
FastCandMatcher<C>::FastCandMatcher( const typename FastCandMatcher<C>::map_type & map ):
  maps_( 1, & map ) {
}

template<typename C>
const reco::Candidate * FastCandMatcher<C>::operator()( const reco::Candidate & c ) const {
  using namespace reco;
  using namespace std;
  if ( c.hasMasterClone() )
    return (*this)( * c.masterClone() );
  unsigned int nDau = c.numberOfDaughters();
  if ( nDau > 0 ) {
    // check for composite candidate c
    // navigate to daughters and find parent matches
    set<const reco::Candidate *> momsIntersection, momDaughters, tmp;
    for( Candidate::const_iterator dau = c.begin(); dau != c.end(); ++ dau ) {
      // check here generically if status == 3, then descend down to one more level
      const Candidate * dauMatch = (*this)( * dau );
      // if a daughter does not match, return a null ref.
      if ( dauMatch == 0 ) return 0;
      // get matched mothers
      size_t mothers = dauMatch->numberOfMothers();
      for( size_t i = 0; i < mothers; ++ i ) {
	const reco::Candidate * mom = dauMatch->mother( i );
	if ( mom != 0 && mom->pdgId() == dauMatch->pdgId() && 
	     mom->status() == 3 && dauMatch->status() == 1 ) {
	  // assume a single mother at this point...
	  mom = mom->mother( 0 );
	}
	momDaughters.insert( mom );
      }
      // if no mother was found return null reference
      if ( momDaughters.size() == 0 ) return 0;
      // the first time, momsIntersection is set to momDaughters
      if ( momsIntersection.size() == 0 ) momsIntersection = momDaughters;
      else {
	tmp.clear();
	set_intersection( momsIntersection.begin(), momsIntersection.end(),
			  momDaughters.begin(), momDaughters.end(),
			 inserter( tmp, tmp.begin() ) );
	swap( momsIntersection, tmp );
      }
      if ( momsIntersection.size() == 0 ) return 0;
    }
    // if multiple mothers are found, return a null reference
    if ( momsIntersection.size() > 1 ) return 0;
    // return a reference to the unique mother
    return * momsIntersection.begin();
  } else {
    // check for non-composite (leaf) candidate 
    // if one of the maps contains the candidate c
    for( typename std::vector<const map_type *>::const_iterator m = maps_.begin(); 
	 m != maps_.end(); ++ m ) {
      for( typename map_type::const_iterator i = (*m)->begin(); i != (*m)->end(); ++ i ) {
	if ( & * i->key == & c )
	  return & * i->val;
      }
    }
    return 0;
  }
}

#endif
