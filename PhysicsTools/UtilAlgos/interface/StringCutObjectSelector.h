#ifndef UtilAlgos_StringCutObjectSelector_h
#define UtilAlgos_StringCutObjectSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Parser/interface/StringCutObjectSelector.h"

namespace reco {
  namespace modules {
    
    template<typename T>
    struct ParameterAdapter<StringCutObjectSelector<T> > {
      static StringCutObjectSelector<T> make( const edm::ParameterSet & cfg ) {
	return StringCutObjectSelector<T>( cfg.template getParameter<std::string>( "cut" ) );
      }
    };
    
  }
}

#endif
