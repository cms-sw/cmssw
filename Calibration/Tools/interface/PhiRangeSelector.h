#ifndef UtilAlgos_PhiRangeSelector_h
#define UtilAlgos_PhiRangeSelector_h
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

struct PhiRangeSelector {
  PhiRangeSelector(double phiMin, double phiMax) : phiMin_(phiMin), phiMax_(phiMax) {}
  template <typename T>
  bool operator()(const T& t) const {
    double phi = t.phi();
    return (phi >= phiMin_ && phi <= phiMax_);
  }

private:
  double phiMin_, phiMax_;
};

namespace reco {
  namespace modules {
    template <>
    struct ParameterAdapter<PhiRangeSelector> {
      static PhiRangeSelector make(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC) {
        return PhiRangeSelector(cfg.getParameter<double>("phiMin"), cfg.getParameter<double>("phiMax"));
      }

      static void fillPSetDescription(edm::ParameterSetDescription& desc) {
        desc.add<double>("phiMin", -3.2);
        desc.add<double>("phiMax", 3.2);
      }
    };
  }  // namespace modules
}  // namespace reco

#endif
