#ifndef UtilAlgos_AndSelector_h
#define UtilAlgos_AndSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/AndSelector.h"

namespace reco {
  namespace modules {
    
    template<typename S1, typename S2>
    struct ParameterAdapter<AndSelector<S1, S2> > {
      static AndSelector<S1, S2> make( const edm::ParameterSet & cfg ) {
	return AndSelector<S1, S2>( modules::make<S1>( cfg ), modules::make<S2>( cfg ) ); 
      }
    };

  }
}

#endif
