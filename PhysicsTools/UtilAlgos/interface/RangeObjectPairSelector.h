#ifndef UtilAlgos_RangeObjectPairSelector_h
#define UtilAlgos_RangeObjectPairSelector_h
#include "PhysicsTools/Utilities/interface/RangeObjectPairSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"

namespace reco {
  namespace modules {
    
    template<typename T, typename F>
    struct ParameterAdapter<RangeObjectPairSelector<T, F> > {
      static RangeObjectPairSelector<T, F> make( const edm::ParameterSet & cfg ) {
	return RangeObjectPairSelector<T, F>( cfg.template getParameter<double>( "rangeMin" ),
					      cfg.template getParameter<double>( "rangeMax" ),
					      reco::modules::make<F>( cfg )
					      );
      }
    };
    
  }
}



#endif
