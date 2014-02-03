#ifndef UtilAlgos_MaxSelector_h
#define UtilAlgos_MaxSelector_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/MaxSelector.h"

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

