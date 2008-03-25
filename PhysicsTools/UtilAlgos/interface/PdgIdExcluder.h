#ifndef PhysicsTools_UtilAlgos_PdgIdSelector_h
#define PhysicsTools_UtilAlgos_PdgIdSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/PdgIdExcluder.h"

namespace reco {
  namespace modules {
    
    template<>
      struct ParameterAdapter<PdgIdExcluder> {
	static PdgIdExcluder make( const edm::ParameterSet & cfg ) {
	  return PdgIdExcluder( cfg.getParameter<std::vector<int> >( "pdgId" ) );
	}
      };

  }
}

#endif
