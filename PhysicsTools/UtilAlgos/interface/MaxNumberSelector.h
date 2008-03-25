#ifndef UtilAlgos_MaxNumberSelector_h
#define UtilAlgos_MaxNumberSelector_h
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "PhysicsTools/Utilities/interface/MaxNumberSelector.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<MaxNumberSelector> {
      static MaxNumberSelector make(const edm::ParameterSet & cfg) {
	return MaxNumberSelector(cfg.getParameter<unsigned int>("maxNumber"));
      }
    };

  }
}

#endif
