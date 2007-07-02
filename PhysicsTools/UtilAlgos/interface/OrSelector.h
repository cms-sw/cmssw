#ifndef UtilAlgos_OrSelector_h
#define UtilAlgos_OrSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/OrSelector.h"

namespace reco {
  namespace modules {
    
    template<typename S1, typename S2>
    struct ParameterAdapter<OrSelector<S1, S2> > {
      static OrSelector<S1, S2> make( const edm::ParameterSet & cfg ) {
	return OrSelector<S1, S2>( modules::make<S1>( cfg ), modules::make<S2>( cfg ) ); 
      }
    };

  }
}

#endif
