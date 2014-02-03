#ifndef UtilAlgos_MinNumberSelector_h
#define UtilAlgos_MinNumberSelector_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/MinNumberSelector.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<MinNumberSelector> {
      static MinNumberSelector make(const edm::ParameterSet & cfg) {
	return MinNumberSelector(cfg.getParameter<unsigned int>("minNumber"));
      }
    };

  }
}

#endif

