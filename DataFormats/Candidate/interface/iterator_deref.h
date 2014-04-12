#ifndef Candidate_iterator_deref_h
#define Candidate_iterator_deref_h

#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Ptr.h"
#include <boost/static_assert.hpp>
#include <vector>

namespace reco {
  namespace candidate {
    
    template<typename C>
    struct iterator_deref {
      BOOST_STATIC_ASSERT(sizeof(C) == 0); 
    };

    template<typename T>
    struct iterator_deref<std::vector<T> > {
      static const Candidate & deref( const typename std::vector<T>::const_iterator & i ) {
	return * i;
      }
    };

    template<typename T>
    struct iterator_deref<edm::OwnVector<T> > {
      static const Candidate & deref( const typename edm::OwnVector<T>::const_iterator & i ) {
	return * i;
      }
    };

    template<typename C>
    struct iterator_deref<edm::RefVector<C> > {
      static const Candidate & deref( const typename edm::RefVector<C>::const_iterator & i ) {
	return * * i;
      }
    };    

    template<typename T>
    struct iterator_deref<std::vector<edm::RefToBase<T> > > {
      static const Candidate & deref( const typename std::vector<edm::RefToBase<T> >::const_iterator & i ) {
	return * * i;
      }
    };    

    template<typename T>
    struct iterator_deref<std::vector<edm::Ptr<T> > > {
      static const Candidate & deref( const typename std::vector<edm::Ptr<T> >::const_iterator & i ) {
	return * * i;
      }
    };    
  }
}

#endif
