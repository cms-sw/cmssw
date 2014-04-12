#ifndef UtilAlgos_MinSelector_h
#define UtilAlgos_MinSelector_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/MinSelector.h"

namespace reco {
  namespace modules {
    
    template<typename T>
    struct ParameterAdapter<MinSelector<T> > {
      static MinSelector<T> make( const edm::ParameterSet & cfg ) {
	return MinSelector<T>( cfg.template getParameter<double>( "min" ) );
      }
    };

  }
}

#endif

