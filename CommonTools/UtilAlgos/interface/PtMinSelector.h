#ifndef UtilAlgos_PtMinSelector_h
#define UtilAlgos_PtMinSelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/PtMinSelector.h"

namespace reco {
  namespace modules {

    template <>
    struct ParameterAdapter<PtMinSelector> {
      static PtMinSelector make(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC) {
        return PtMinSelector(cfg.getParameter<double>("ptMin"));
      }

      static void fillPSetDescription(edm::ParameterSetDescription& desc) { desc.add<double>("ptMin", 0.); }
    };

  }  // namespace modules
}  // namespace reco

#endif
