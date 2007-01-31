#ifndef UtilAlgos_DeltaR_h
#define UtilAlgos_DeltaR_h
#include "PhysicsTools/Utilities/interface/DeltaR.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"

namespace reco {
  namespace modules {
    
    template<typename T>
    struct ParameterAdapter<DeltaR<T> > {
      static DeltaR<T> make( const edm::ParameterSet & cfg ) {
	return DeltaR<T>();
      }
    };
    
  }
}



#endif
