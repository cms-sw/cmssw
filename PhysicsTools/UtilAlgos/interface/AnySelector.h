#ifndef UtilAlgos_AnySelector_h
#define UtilAlgos_AnySelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/AnySelector.h"

namespace reco {
  namespace modules {
    
    template<typename T>
    struct ParameterAdapter<AnySelector<T> > {
      static AnySelector<T> make( const edm::ParameterSet & cfg ) {
	return AnySelector<T>();
      }
    };

  }
}

#endif
