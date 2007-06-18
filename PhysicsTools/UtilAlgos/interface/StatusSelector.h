#ifndef UtilAlgos_StatusSelector_h
#define UtilAlgos_StatusSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/StatusSelector.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<StatusSelector> {
      static StatusSelector make( const edm::ParameterSet & cfg ) {
	return StatusSelector( cfg.getParameter<std::vector<int> >( "status" ) );
      }
    };

  }
}

#endif
