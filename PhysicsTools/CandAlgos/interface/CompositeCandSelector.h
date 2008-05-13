#ifndef PhysicsTools_CandAlgos_CompositeCandSelector_h
#define PhysicsTools_CandAlgos_CompositeCandSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/CandUtils/interface/CompositeCandSelector.h"

namespace reco {
  namespace modules {
    
    template<typename Selector, typename T1, typename T2, unsigned int nDau>
      struct ParameterAdapter<CompositeCandSelector<Selector, T1, T2, nDau> > {
	static CompositeCandSelector<Selector, T1, T2, nDau> make(const edm::ParameterSet & cfg) {
	  return CompositeCandSelector<Selector, T1, T2, nDau>(modules::make<Selector>(cfg));
	}
      };

  }
}

#endif
