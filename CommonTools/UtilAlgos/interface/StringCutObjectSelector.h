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


// Introducing a simpler way to use a string object selector outside of the 
// heavily-templated infrastructure above. This simply translates the cfg
// into the string that the functor expects. 
namespace reco{
  template< typename T, bool Lazy>
    class StringCutObjectSelectorHandler : public StringCutObjectSelector<T,Lazy>  {
    public:
      explicit StringCutObjectSelectorHandler( const edm::ParameterSet & cfg ) :
        StringCutObjectSelector<T, Lazy>(cfg.getParameter<std::string>("cut"))
      {
      }
  };
}

#endif

