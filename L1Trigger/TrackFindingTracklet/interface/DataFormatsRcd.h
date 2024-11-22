#ifndef L1Trigger_TrackFindingTracklet_DataFormatsRcd_h
#define L1Trigger_TrackFindingTracklet_DataFormatsRcd_h

#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "L1Trigger/TrackFindingTracklet/interface/ChannelAssignmentRcd.h"
#include "FWCore/Utilities/interface/mplVector.h"

namespace trklet {

  typedef edm::mpl::Vector<ChannelAssignmentRcd> RcdsDataFormats;

  // record of trklet::DataFormats
  class DataFormatsRcd : public edm::eventsetup::DependentRecordImplementation<DataFormatsRcd, RcdsDataFormats> {};

}  // namespace trklet

#endif