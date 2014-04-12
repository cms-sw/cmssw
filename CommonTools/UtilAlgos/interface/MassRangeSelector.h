#ifndef UtilAlgos_MassRangeSelector_h
#define UtilAlgos_MassRangeSelector_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/MassRangeSelector.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<MassRangeSelector> {
      static MassRangeSelector make( const edm::ParameterSet & cfg ) {
	return 
	  MassRangeSelector( cfg.getParameter<double>( "massMin" ),
			     cfg.getParameter<double>( "massMax" ) );
      }
    };
    
  }
}

#endif

