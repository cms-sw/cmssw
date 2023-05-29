#ifndef L1Trigger_TrackerTFP_DemonstratorRcd_h
#define L1Trigger_TrackerTFP_DemonstratorRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "L1Trigger/TrackTrigger/interface/SetupRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

namespace trackerTFP {

  typedef edm::mpl::Vector<tt::SetupRcd> RcdsDemonstrator;

  // record of trackerTFP::Demonstrator
  class DemonstratorRcd : public edm::eventsetup::DependentRecordImplementation<DemonstratorRcd, RcdsDemonstrator> {};

}  // namespace trackerTFP

#endif