#ifndef UtilAlgos_MasslessInvariantMass_h
#define UtilAlgos_MasslessInvariantMass_h
#include "PhysicsTools/Utilities/interface/MasslessInvariantMass.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"

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
