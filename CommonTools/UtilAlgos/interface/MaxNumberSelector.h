#ifndef UtilAlgos_MaxNumberSelector_h
#define UtilAlgos_MaxNumberSelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/MaxNumberSelector.h"

namespace reco {
  namespace modules {

    template <>
    struct ParameterAdapter<MaxNumberSelector> {
      static MaxNumberSelector make(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC) {
        return MaxNumberSelector(cfg.getParameter<unsigned int>("maxNumber"));
      }

      static void fillPSetDescription(edm::ParameterSetDescription& desc) { desc.add<unsigned int>("maxNumber", 0); }
    };

  }  // namespace modules
}  // namespace reco

#endif
