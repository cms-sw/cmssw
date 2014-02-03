#ifndef UtilAlgos_ParameterAdapter_h
#define UtilAlgos_ParameterAdapter_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace reco {
  namespace modules {
    
    template<typename S> 
    struct ParameterAdapter { 
      static S make(const edm::ParameterSet & cfg) {
	return S(cfg);
      }
    };
    
    template<typename S>
    S make(const edm::ParameterSet & cfg) {
      return ParameterAdapter<S>::make(cfg);
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
    }; \
  } \
}

#endif

