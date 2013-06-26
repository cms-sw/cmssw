#ifndef UtilAlgos_ChargeSelector_h
#define UtilAlgos_ChargeSelector_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/ChargeSelector.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<ChargeSelector> {
      static ChargeSelector make( const edm::ParameterSet & cfg ) {
	return ChargeSelector( cfg.getParameter<int>( "charge" ) );
      }
    };

  }
}

#endif

