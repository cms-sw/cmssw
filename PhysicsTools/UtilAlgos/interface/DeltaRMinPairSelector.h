#ifndef UtilAlgos_DeltaRMinPairSelector_h
#define UtilAlgos_DeltaRMinPairSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/DeltaRMinPairSelector.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<DeltaRMinPairSelector> {
      static DeltaRMinPairSelector make( const edm::ParameterSet & cfg ) {
	return DeltaRMinPairSelector( cfg.getParameter<double>( "deltaRMin" ) );
      }
    };

  }
}

#endif
