#include "DataFormats/Common/interface/Wrapper.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

namespace RecoTracker_MeasurementDet {
  struct dictionary {
    MeasurementTrackerEvent dummy;
    edm::Wrapper<MeasurementTrackerEvent> dummy1;
  };
}  // namespace RecoTracker_MeasurementDet
