#ifndef UtilAlgos_EtaRangeSelector_h
#define UtilAlgos_EtaRangeSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/EtaRangeSelector.h"

namespace reco {
  namespace modules {
    
    template<typename T>
    struct ParameterAdapter<EtaRangeSelector<T> > {
      static EtaRangeSelector<T> make( const edm::ParameterSet & cfg ) {
	return 
	  EtaRangeSelector<T>( cfg.template getParameter<double>( "etaMin" ),
			       cfg.template getParameter<double>( "etaMax" ) );
      }
    };

  }
}

#endif
