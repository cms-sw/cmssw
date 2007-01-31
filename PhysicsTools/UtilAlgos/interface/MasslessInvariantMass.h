#ifndef UtilAlgos_MasslessInvariantMass_h
#define UtilAlgos_MasslessInvariantMass_h
#include "PhysicsTools/Utilities/interface/MasslessInvariantMass.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"

namespace reco {
  namespace modules {
    
    template<typename T>
    struct ParameterAdapter<MasslessInvariantMass<T> > {
      static MasslessInvariantMass<T> make( const edm::ParameterSet & cfg ) {
	return MasslessInvariantMass<T>();
      }
    };
    
  }
}



#endif
