#ifndef Candidate_iterator_deref_h
#define Candidate_iterator_deref_h

#include "DataFormats/Candidate/interface/Candidate.h"
#include<boost/static_assert.hpp>

namespace reco {
  namespace candidate {
    
    template<typename C>
    struct iterator_deref {
      BOOST_STATIC_ASSERT(sizeof(C) == 0); 
    };

    template<>
    struct iterator_deref<CandidateCollection> {
      static const Candidate & deref( const CandidateCollection::const_iterator & i ) {
	return * i;
      }
    };

    template<>
    struct iterator_deref<CandidateRefVector> {
      static const Candidate & deref( const CandidateRefVector::const_iterator & i ) {
	return * * i;
      }
    };    

    template<>
    struct iterator_deref<std::vector<CandidateBaseRef> > {
      static const Candidate & deref( const std::vector<CandidateBaseRef>::const_iterator & i ) {
	return * * i;
      }
    };    
  }
}

#endif
