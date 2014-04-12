#ifndef UtilAlgos_DeltaPhiMinPairSelector_h
#define UtilAlgos_DeltaPhiMinPairSelector_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/DeltaPhiMinPairSelector.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<DeltaPhiMinPairSelector> {
      static DeltaPhiMinPairSelector make( const edm::ParameterSet & cfg ) {
	return DeltaPhiMinPairSelector( cfg.getParameter<double>( "deltaPhiMin" ) );
      }
    };

  }
}

#endif

