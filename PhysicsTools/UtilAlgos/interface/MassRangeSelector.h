#ifndef UtilAlgos_MassRangeSelector_h
#define UtilAlgos_MassRangeSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/MassRangeSelector.h"

namespace reco {
  namespace modules {
    
    template<typename T>
    struct ParameterAdapter<MassRangeSelector<T> > {
      static MassRangeSelector<T> make( const edm::ParameterSet & cfg ) {
	return 
	  MassRangeSelector<T>( cfg.template getParameter<double>( "massMin" ),
				cfg.template getParameter<double>( "massMax" ) );
      }
    };
    
  }
}

#endif
