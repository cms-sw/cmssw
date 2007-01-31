#ifndef UtilAlgos_MinNumberSelector_h
#define UtilAlgos_MinNumberSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/MinNumberSelector.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<MinNumberSelector> {
      static MinNumberSelector make( const edm::ParameterSet & cfg ) {
	return MinNumberSelector( cfg.getParameter<unsigned int>( "minNumber" ) );
      }
    };

  }
}

#endif
