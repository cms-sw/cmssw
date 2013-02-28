#ifndef UtilAlgos_AnyPairSelector_h
#define UtilAlgos_AnyPairSelector_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/AnyPairSelector.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<AnyPairSelector> {
      static AnyPairSelector make( const edm::ParameterSet & cfg ) {
	return AnyPairSelector();
      }
    };

  }
}

#endif

