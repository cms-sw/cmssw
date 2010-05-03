#ifndef UtilAlgos_DeltaR_h
#define UtilAlgos_DeltaR_h
#include "CommonTools/Utils/interface/deltaR.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"

namespace reco {
  namespace modules {
    
    template<typename T1, typename T2>
    struct ParameterAdapter<DeltaR<T1, T2> > {
      static DeltaR<T1, T2> make( const edm::ParameterSet & cfg ) {
	return DeltaR<T1, T2>();
      }
    };
    
  }
}



#endif
