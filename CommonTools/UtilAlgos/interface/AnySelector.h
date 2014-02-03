#ifndef UtilAlgos_AnySelector_h
#define UtilAlgos_AnySelector_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/AnySelector.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<AnySelector> {
      static AnySelector make( const edm::ParameterSet & cfg ) {
	return AnySelector();
      }
    };

  }
}

#endif

