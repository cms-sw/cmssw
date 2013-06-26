#ifndef UtilAlgos_DeltaR_h
#define UtilAlgos_DeltaR_h
#include "DataFormats/Math/interface/deltaR.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"

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

