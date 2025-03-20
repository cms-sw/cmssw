#ifndef TrackingForConversion_TrackChargeSelector_h
#define TrackingForConversion_TrackChargeSelector_h
/* \class TrackChargeSelector
 *
 * \author Domenico Giordano, CERN
 *
 */

struct TrackChargeSelector {
  TrackChargeSelector(int charge) : charge_(charge) {}
  template <typename T>
  bool operator()(const T& t) const {
    return (t.charge() == charge_);
  }

private:
  int charge_;
};

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

namespace reco {
  namespace modules {

    template <>
    struct ParameterAdapter<TrackChargeSelector> {
      static TrackChargeSelector make(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC) {
        return TrackChargeSelector(cfg.getParameter<int>("charge"));
      }

      static void fillPSetDescription(edm::ParameterSetDescription& desc) { desc.add<int>("charge", 0); }
    };

  }  // namespace modules
}  // namespace reco

#endif
