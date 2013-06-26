#ifndef UtilAlgos_AndPairSelector_h
#define UtilAlgos_AndPairSelector_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/AndPairSelector.h"

namespace reco {
  namespace modules {
    
    template<typename S1, typename S2>
    struct ParameterAdapter<AndPairSelector<S1, S2> > {
      static AndPairSelector<S1, S2> make(const edm::ParameterSet & cfg) {
	return AndPairSelector<S1, S2>(modules::make<S1>(cfg.getParameter<edm::ParameterSet>("cut1")), 
				       modules::make<S2>(cfg.getParameter<edm::ParameterSet>("cut2"))); 
      }
    };

  }
}

#endif

