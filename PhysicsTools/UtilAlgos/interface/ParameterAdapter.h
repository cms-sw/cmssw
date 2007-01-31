#ifndef UtilAlgos_ParameterAdapter_h
#define UtilAlgos_ParameterAdapter_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <boost/static_assert.hpp>

namespace reco {
  namespace modules {
    
    template<typename S> 
    struct ParameterAdapter { 
      BOOST_STATIC_ASSERT( sizeof( S ) == 0 ); 
    };
    
    template<typename S>
    S make( const edm::ParameterSet & cfg ) {
      return ParameterAdapter<S>::make( cfg );
    }

  }
}

#endif
