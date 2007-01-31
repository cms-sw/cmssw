#ifndef UtilAlgos_AnyPairSelector_h
#define UtilAlgos_AnyPairSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/AnyPairSelector.h"

namespace reco {
  namespace modules {
    
    template<typename T>
    struct ParameterAdapter<AnyPairSelector<T> > {
      static AnyPairSelector<T> make( const edm::ParameterSet & cfg ) {
	return AnyPairSelector<T>();
      }
    };

  }
}

#endif
