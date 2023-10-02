#ifndef UtilAlgos_DetSetCounterSelector_h
#define UtilAlgos_DetSetCounterSelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "CommonTools/Utils/interface/DetSetCounterSelector.h"

namespace reco {
  namespace modules {

    template <>
    struct ParameterAdapter<DetSetCounterSelector> {
      static DetSetCounterSelector make(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC) {
        return DetSetCounterSelector(cfg.getParameter<unsigned int>("minDetSetCounts"),
                                     cfg.getParameter<unsigned int>("maxDetSetCounts"));
      }

      static void fillPSetDescription(edm::ParameterSetDescription& desc) {
        desc.add<unsigned int>("minDetSetCounts", 0);
        desc.add<unsigned int>("maxDetSetCounts", 0);
      }
    };

  }  // namespace modules
}  // namespace reco

#endif
