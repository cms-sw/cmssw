#ifndef L1Trigger_TrackerDTC_LayerEncodingRcd_h
#define L1Trigger_TrackerDTC_LayerEncodingRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "L1Trigger/TrackTrigger/interface/SetupRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

namespace trackerDTC {

  typedef edm::mpl::Vector<tt::SetupRcd> RcdsLayerEncoding;

  // record of trackerDTC::LayerEncoding
  class LayerEncodingRcd : public edm::eventsetup::DependentRecordImplementation<LayerEncodingRcd, RcdsLayerEncoding> {
  };

}  // namespace trackerDTC

#endif