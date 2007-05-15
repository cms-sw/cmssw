#ifndef UtilAlgos_MaxSelector_h
#define UtilAlgos_MaxSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/MaxSelector.h"

namespace reco {
  namespace modules {
    
    template<typename T>
    struct ParameterAdapter<MaxSelector<T> > {
      static MaxSelector<T> make( const edm::ParameterSet & cfg ) {
	return MaxSelector<T>( cfg.template getParameter<double>( "max" ) );
      }
    };

  }
}

#endif
