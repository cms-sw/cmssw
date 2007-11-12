#ifndef UtilAlgos_MasslessInvariantMass_h
#define UtilAlgos_MasslessInvariantMass_h
#include "PhysicsTools/Utilities/interface/MasslessInvariantMass.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"

namespace reco {
  namespace modules {
    
    template<typename T1, typename T2>
    struct ParameterAdapter<MasslessInvariantMass<T1, T2> > {
      static MasslessInvariantMass<T1, T2> make( const edm::ParameterSet & cfg ) {
	return MasslessInvariantMass<T1, T2>();
      }
    };
    
  }
}



#endif
