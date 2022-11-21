#ifndef L1Trigger_TrackerTFP_KalmanFilterFormatsRcd_h
#define L1Trigger_TrackerTFP_KalmanFilterFormatsRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "L1Trigger/TrackerTFP/interface/DataFormatsRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

namespace trackerTFP {

  typedef edm::mpl::Vector<DataFormatsRcd> RcdsKalmanFilterFormats;

  class KalmanFilterFormatsRcd
      : public edm::eventsetup::DependentRecordImplementation<KalmanFilterFormatsRcd, RcdsKalmanFilterFormats> {};

}  // namespace trackerTFP

#endif