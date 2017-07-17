#ifndef UtilAlgos_MaxNumberSelector_h
#define UtilAlgos_MaxNumberSelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/MaxNumberSelector.h"

namespace reco {
  namespace modules {

    template<>
    struct ParameterAdapter<MaxNumberSelector> {
      static MaxNumberSelector make(const edm::ParameterSet & cfg, edm::ConsumesCollector & iC) {
	return MaxNumberSelector(cfg.getParameter<unsigned int>("maxNumber"));
      }
    };

  }
}

#endif

