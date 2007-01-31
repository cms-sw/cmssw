#ifndef UtilAlgos_SingleObjectSelector_h
#define UtilAlgos_SingleObjectSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Parser/interface/SingleObjectSelector.h"

namespace reco {
  namespace modules {
    
    template<typename T>
    struct ParameterAdapter<SingleObjectSelector<T> > {
      static SingleObjectSelector<T> make( const edm::ParameterSet & cfg ) {
	return SingleObjectSelector<T>( cfg.template getParameter<std::string>( "cut" ) );
      }
    };
    
  }
}

#endif
