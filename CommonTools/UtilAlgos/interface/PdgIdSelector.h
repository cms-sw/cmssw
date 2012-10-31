#ifndef UtilAlgos_PdgIdSelector_h
#define UtilAlgos_PdgIdSelector_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/PdgIdSelector.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<PdgIdSelector> {
      static PdgIdSelector make( const edm::ParameterSet & cfg ) {
	return PdgIdSelector( cfg.getParameter<std::vector<int> >( "pdgId" ) );
      }
    };

  }
}

#endif
