#ifndef UtilAlgos_StatusSelector_h
#define UtilAlgos_StatusSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/StatusSelector.h"

namespace reco {
  namespace modules {
    
    template<typename T>
    struct ParameterAdapter<StatusSelector<T> > {
      static StatusSelector<T> make( const edm::ParameterSet & cfg ) {
	return StatusSelector<T>( cfg.template getParameter<std::vector<int> >( "status" ) );
      }
    };

  }
}

#endif
