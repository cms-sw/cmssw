#ifndef UtilAlgos_PdgIdSelector_h
#define UtilAlgos_PdgIdSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/PdgIdSelector.h"

namespace reco {
  namespace modules {
    
    template<typename T>
    struct ParameterAdapter<PdgIdSelector<T> > {
      static PdgIdSelector<T> make( const edm::ParameterSet & cfg ) {
	return PdgIdSelector<T>( cfg.template getParameter<std::vector<int> >( "pdgId" ) );
      }
    };

  }
}

#endif
