#ifndef L1Trigger_TrackerTFP_DataFormatsRcd_h
#define L1Trigger_TrackerTFP_DataFormatsRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "L1Trigger/TrackTrigger/interface/SetupRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

namespace trackerTFP {

  typedef edm::mpl::Vector<tt::SetupRcd> RcdsDataFormats;

  // record of trackerTFP::DataFormats
  class DataFormatsRcd : public edm::eventsetup::DependentRecordImplementation<DataFormatsRcd, RcdsDataFormats> {};

}  // namespace trackerTFP

#endif