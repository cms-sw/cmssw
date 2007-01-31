#ifndef UtilAlgos_PtMinSelector_h
#define UtilAlgos_PtMinSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/PtMinSelector.h"

namespace reco {
  namespace modules {
    
    template<typename T>
    struct ParameterAdapter<PtMinSelector<T> > {
      static PtMinSelector<T> make( const edm::ParameterSet & cfg ) {
	return PtMinSelector<T>( cfg.template getParameter<double>( "ptMin" ) );
      }
    };

  }
}

#endif
