#ifndef UtilAlgos_EtMinSelector_h
#define UtilAlgos_EtMinSelector_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/EtMinSelector.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<EtMinSelector> {
      static EtMinSelector make( const edm::ParameterSet & cfg ) {
	return EtMinSelector( cfg.getParameter<double>( "etMin" ) );
      }
    };

  }
}

#endif

