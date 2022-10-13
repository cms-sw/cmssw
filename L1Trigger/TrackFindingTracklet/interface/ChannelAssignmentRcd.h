#ifndef L1Trigger_TrackFindingTracklet_ChannelAssignmentRcd_h
#define L1Trigger_TrackFindingTracklet_ChannelAssignmentRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"
#include "L1Trigger/TrackTrigger/interface/SetupRcd.h"

namespace trklet {

  typedef edm::mpl::Vector<tt::SetupRcd> RcdsChannelAssignment;

  class ChannelAssignmentRcd
      : public edm::eventsetup::DependentRecordImplementation<ChannelAssignmentRcd, RcdsChannelAssignment> {};

}  // namespace trklet

#endif