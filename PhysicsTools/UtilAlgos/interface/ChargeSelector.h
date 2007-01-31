#ifndef UtilAlgos_ChargeSelector_h
#define UtilAlgos_ChargeSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/ChargeSelector.h"

namespace reco {
  namespace modules {
    
    template<typename T>
    struct ParameterAdapter<ChargeSelector<T> > {
      static ChargeSelector<T> make( const edm::ParameterSet & cfg ) {
	return ChargeSelector<T>( cfg.template getParameter<int>( "charge" ) );
      }
    };

  }
}

#endif
