#ifndef L1Trigger_TrackFindingTracklet_TrackBuilderChannelRcd_h
#define L1Trigger_TrackFindingTracklet_TrackBuilderChannelRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "FWCore/Utilities/interface/mplVector.h"
#include "L1Trigger/TrackTrigger/interface/SetupRcd.h"

namespace trackFindingTracklet {

  typedef edm::mpl::Vector<tt::SetupRcd> RcdsTrackBuilderChannel;

  class TrackBuilderChannelRcd
      : public edm::eventsetup::DependentRecordImplementation<TrackBuilderChannelRcd, RcdsTrackBuilderChannel> {};

}  // namespace trackFindingTracklet

#endif