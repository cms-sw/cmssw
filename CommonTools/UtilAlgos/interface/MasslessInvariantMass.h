#ifndef UtilAlgos_MasslessInvariantMass_h
#define UtilAlgos_MasslessInvariantMass_h
#include "CommonTools/Utils/interface/MasslessInvariantMass.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<MasslessInvariantMass> {
      static MasslessInvariantMass make( const edm::ParameterSet & cfg ) {
	return MasslessInvariantMass();
      }
    };
    
  }
}



#endif

