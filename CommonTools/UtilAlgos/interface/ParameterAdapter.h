#ifndef UtilAlgos_ParameterAdapter_h
#define UtilAlgos_ParameterAdapter_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace reco {
  namespace modules {

    template <typename S>
    struct ParameterAdapter {
      static S make(const edm::ParameterSet& cfg) { return S(cfg); }
      static S make(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC) { return S(cfg, iC); }
      static S make(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC) { return S(cfg, iC); }

      static void fillPSetDescription(edm::ParameterSetDescription& desc) { S::template fillPSetDescription(desc); }
    };

    template <typename S>
    S make(const edm::ParameterSet& cfg) {
      return ParameterAdapter<S>::make(cfg);
    }
    template <typename S>
    S make(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC) {
      return ParameterAdapter<S>::make(cfg, iC);
    }
    template <typename S>
    S make(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC) {
      return ParameterAdapter<S>::make(cfg, iC);
    }

    template <typename S>
    void fillPSetDescription(edm::ParameterSetDescription& desc) {
      ParameterAdapter<S>::fillPSetDescription(desc);
    }
  }  // namespace modules
}  // namespace reco

#define NOPARAMETER_ADAPTER(TYPE)                                                                      \
  namespace reco {                                                                                     \
    namespace modules {                                                                                \
      struct ParameterAdapter<TYPE> {                                                                  \
        static TYPE make(const edm::ParameterSet& cfg) { return TYPE(); }                              \
        static TYPE make(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC) { return TYPE(); } \
        static TYPE make(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC) { return TYPE(); }  \
        static void fillPSetDescription(edm::ParameterSetDescription& desc) {}                         \
      };                                                                                               \
    }                                                                                                  \
  }

#endif
