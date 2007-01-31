#ifndef UtilAlgos_NonNullNumberSelector_h
#define UtilAlgos_NonNullNumberSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/NonNullNumberSelector.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<NonNullNumberSelector> {
      static NonNullNumberSelector make( const edm::ParameterSet & cfg ) {
	return NonNullNumberSelector();
      }
    };

  }
}

#endif
