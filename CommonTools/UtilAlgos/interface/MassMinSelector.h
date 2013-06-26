#ifndef UtilAlgos_MassMinSelector_h
#define UtilAlgos_MassMinSelector_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/MassMinSelector.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<MassMinSelector> {
      static MassMinSelector make( const edm::ParameterSet & cfg ) {
	return 
	  MassMinSelector( cfg.getParameter<double>( "massMin" ) );
      }
    };
    
  }
}

#endif

