//
//  Created by J.Li on 1/23/21.
//

#ifndef L1Trigger_TrackTrigger_interface_HitPatternHelperRcd_h
#define L1Trigger_TrackTrigger_interface_HitPatternHelperRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "L1Trigger/TrackTrigger/interface/SetupRcd.h"
#include "L1Trigger/TrackerTFP/interface/DataFormatsRcd.h"
#include "L1Trigger/TrackerTFP/interface/LayerEncodingRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

namespace hph {

  typedef edm::mpl::Vector<tt::SetupRcd, trackerTFP::DataFormatsRcd, trackerTFP::LayerEncodingRcd> Rcds;

  // record of hph::SetupRcd
  class SetupRcd : public edm::eventsetup::DependentRecordImplementation<SetupRcd, Rcds> {};

}  // namespace hph

#endif
