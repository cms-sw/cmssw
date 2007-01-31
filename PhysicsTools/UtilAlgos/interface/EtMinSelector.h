#ifndef UtilAlgos_EtMinSelector_h
#define UtilAlgos_EtMinSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/EtMinSelector.h"

namespace reco {
  namespace modules {
    
    template<typename T>
    struct ParameterAdapter<EtMinSelector<T> > {
      static EtMinSelector<T> make( const edm::ParameterSet & cfg ) {
	return EtMinSelector<T>( cfg.template getParameter<double>( "etMin" ) );
      }
    };

  }
}

#endif
