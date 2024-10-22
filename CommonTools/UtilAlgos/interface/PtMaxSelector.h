#ifndef UtilAlgos_PtMaxSelector_h
#define UtilAlgos_PtMaxSelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/PtMaxSelector.h"

namespace reco {
  namespace modules {

    template <>
    struct ParameterAdapter<PtMaxSelector> {
      static PtMaxSelector make(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC) {
        return PtMaxSelector(cfg.getParameter<double>("ptMax"));
      }

      static void fillPSetDescription(edm::ParameterSetDescription& desc) { desc.add<double>("ptMax", 0.); }
    };

  }  // namespace modules
}  // namespace reco

#endif
