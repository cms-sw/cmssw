#ifndef UtilAlgos_EtMinSelector_h
#define UtilAlgos_EtMinSelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/EtMinSelector.h"

namespace reco {
  namespace modules {

    template <>
    struct ParameterAdapter<EtMinSelector> {
      static EtMinSelector make(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC) {
        return EtMinSelector(cfg.getParameter<double>("etMin"));
      }

      static void fillPSetDescription(edm::ParameterSetDescription& desc) { desc.add<double>("etMin", 0.); }
    };

  }  // namespace modules
}  // namespace reco

#endif
