#ifndef DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsMaskingSoA_h
#define DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsMaskingSoA_h

#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace reco {

  GENERATE_SOA_LAYOUT(TrackingRecHitsMaskingLayout, SOA_COLUMN(uint32_t, recHitMask));

  using TrackingRecHitsMaskingSoA = TrackingRecHitsMaskingLayout<>;
  using TrackingRecHitsMaskingView = TrackingRecHitsMaskingSoA::View;
  using TrackingRecHitsMaskingConstView = TrackingRecHitsMaskingSoA::ConstView;

};  // namespace reco

#endif  // DataFormats_TrackingRecHitSoA_interface_TrackingRecHitsMaskingSoA_h
