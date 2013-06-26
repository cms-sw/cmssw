#ifndef UtilAlgos_RefSelector_h
#define UtilAlgos_RefSelector_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/RefSelector.h"

namespace reco {
  namespace modules {
    
    template<typename S>
    struct ParameterAdapter<RefSelector<S> > {
      static RefSelector<S> make( const edm::ParameterSet & cfg ) {
	return RefSelector<S>( modules::make<S>( cfg ) ); 
      }
    };

  }
}

#endif

