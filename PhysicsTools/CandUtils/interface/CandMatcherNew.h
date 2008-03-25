#ifndef CandUtils_CandMatcherNew_h
#define CandUtils_CandMatcherNew_h
/* class CandMatcher
 *
 * \author Luca Lista, INFN
 *
 */
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Association.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <algorithm>
#include <iterator>
#include <set>

namespace reco {
  namespace utilsNew {

    template<typename C>
    class CandMatcher {
    public:
      /// map type
      typedef edm::Association<C> map_type;
      /// ref type
      typedef typename edm::Association<C>::reference_type reference_type;
      typedef std::vector<const map_type *> map_vector;
      /// constructor
      explicit CandMatcher(const map_vector & maps);
      /// constructor
      explicit CandMatcher(const map_type & map);
      /// destructor
      virtual ~CandMatcher();
      /// get match from transient reference
      reference_type operator[](const reco::Candidate &) const;
      /// reference to matched collection
      typename map_type::refprod_type ref() const { return map_.ref(); }
    protected:
      /// match map at leaf level
      map_type map_;

    private:
    };

    template<typename C>
    CandMatcher<C>::CandMatcher(const typename CandMatcher<C>::map_vector & maps):
      map_() {
      for(typename map_vector::const_iterator i = maps.begin(); i != maps.end(); ++ i) 
	map_ += **i;
    }
    
    template<typename C>
    CandMatcher<C>::CandMatcher(const typename CandMatcher<C>::map_type & map):
      map_(map) {
    }
    
    template<typename C>
    CandMatcher<C>::~CandMatcher() {
    }
    
    template<typename C>
    typename CandMatcher<C>::reference_type CandMatcher<C>::operator[](const reco::Candidate & c) const {
      using namespace reco;
      using namespace std;
      if (c.hasMasterClone()) {
      	CandidateBaseRef master = c.masterClone();
	return master->numberOfDaughters() == 0 ? map_[master] : (*this)[*master];
      }
      size_t nDau = c.numberOfDaughters();
      if(nDau == 0) return reference_type();
      set<size_t> momIdx, common, tmp;
      for(size_t i = 0; i < nDau; ++ i) {
	const Candidate & d = * c.daughter(i);
	reference_type m = (*this)[d];
	if (m.isNull()) return reference_type();
	momIdx.clear();
	while(m->numberOfMothers() == 1) {
	  m = m->motherRef();
	  momIdx.insert(m.key());
	} 
	if(momIdx.size() == 0) return reference_type();
	if (common.size() == 0) common = momIdx;
	else {
	  tmp.clear();
	  set_intersection(common.begin(), common.end(),
			   momIdx.begin(), momIdx.end(),
			   inserter(tmp, tmp.begin()));
	  swap(common, tmp);
	}
	if (common.size() == 0) return reference_type();
      }
      size_t idx = * max_element(common.begin(), common.end());
      return reference_type(map_.ref(), idx);
    }
   
  }
}

#endif
