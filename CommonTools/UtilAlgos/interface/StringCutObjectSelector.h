#ifndef UtilAlgos_StringCutObjectSelector_h
#define UtilAlgos_StringCutObjectSelector_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

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
