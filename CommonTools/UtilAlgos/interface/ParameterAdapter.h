#ifndef UtilAlgos_ParameterAdapter_h
#define UtilAlgos_ParameterAdapter_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace reco {
  namespace modules {

    template<typename S>
    struct ParameterAdapter {
      static S make(const edm::ParameterSet & cfg) {
	return S(cfg);
      }
      static S make(const edm::ParameterSet & cfg, edm::ConsumesCollector && iC) {
	return S(cfg, iC);
      }
      static S make(const edm::ParameterSet & cfg, edm::ConsumesCollector & iC) {
	return S(cfg, iC);
      }
    };

    template<typename S>
    S make(const edm::ParameterSet & cfg) {
      return ParameterAdapter<S>::make(cfg);
    }
    template<typename S>
    S make(const edm::ParameterSet & cfg, edm::ConsumesCollector && iC) {
      return ParameterAdapter<S>::make(cfg, iC);
    }
    template<typename S>
    S make(const edm::ParameterSet & cfg, edm::ConsumesCollector & iC) {
      return ParameterAdapter<S>::make(cfg, iC);
    }

  }
}

#define NOPARAMETER_ADAPTER(TYPE) \
namespace reco { \
  namespace modules { \
    struct ParameterAdapter<TYPE> { \
      static TYPE make(const edm::ParameterSet & cfg) { \
	return TYPE(); \
      } \
      static TYPE make(const edm::ParameterSet & cfg, edm::ConsumesCollector && iC) { \
	return TYPE(); \
      } \
      static TYPE make(const edm::ParameterSet & cfg, edm::ConsumesCollector & iC) { \
	return TYPE(); \
      } \
    }; \
  } \
}

#endif

