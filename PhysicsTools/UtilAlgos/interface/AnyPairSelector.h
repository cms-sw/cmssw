#ifndef UtilAlgos_AnyPairSelector_h
#define UtilAlgos_AnyPairSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/AnyPairSelector.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<AnyPairSelector> {
      static AnyPairSelector make( const edm::ParameterSet & cfg ) {
	return AnyPairSelector();
      }
    };

  }
}

#endif
