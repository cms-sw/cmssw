#ifndef L1Trigger_TrackerTFP_TrackQualityRcd_h
#define L1Trigger_TrackerTFP_TrackQualityRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "L1Trigger/TrackerTFP/interface/DataFormatsRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

namespace trackerTFP {

  typedef edm::mpl::Vector<DataFormatsRcd> RcdsTrackQuality;

  // record of trackerTFP::TrackQuality
  class TrackQualityRcd : public edm::eventsetup::DependentRecordImplementation<TrackQualityRcd, RcdsTrackQuality> {};

}  // namespace trackerTFP

#endif