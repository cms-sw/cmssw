#ifndef UtilAlgos_OrPairSelector_h
#define UtilAlgos_OrPairSelector_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/OrPairSelector.h"

namespace reco {
  namespace modules {
    
    template<typename S1, typename S2>
    struct ParameterAdapter<OrPairSelector<S1, S2> > {
      static OrPairSelector<S1, S2> make(const edm::ParameterSet & cfg) {
	return OrPairSelector<S1, S2>(modules::make<S1>(cfg.getParameter<edm::ParameterSet>("cut1")), 
				      modules::make<S2>(cfg.getParameter<edm::ParameterSet>("cut2"));) 
      }
    };

  }
}

#endif

