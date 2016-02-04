#ifndef UtilAlgos_StringCutObjectSelector_h
#define UtilAlgos_StringCutObjectSelector_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

namespace reco {
  namespace modules {
    
    template<typename T, bool Lazy>
    struct ParameterAdapter<StringCutObjectSelector<T, Lazy> > {
      static StringCutObjectSelector<T, Lazy> make( const edm::ParameterSet & cfg ) {
	return StringCutObjectSelector<T, Lazy>( cfg.template getParameter<std::string>( "cut" ) );
      }
    };
    
  }
}

#endif
