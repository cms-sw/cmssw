#ifndef L1Trigger_TrackFindingTracklet_TrackBuilderChannel_h
#define L1Trigger_TrackFindingTracklet_TrackBuilderChannel_h

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackBuilderChannelRcd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include <vector>

namespace trackFindingTracklet {

  /*! \class  trackFindingTracklet::TrackBuilderChannel
   *  \brief  Class to assign tracklet tracks to channel
   *          based on ther Pt or seed type,
   *  \author Thomas Schuh
   *  \date   2020, Nov
   */
  class TrackBuilderChannel {
  public:
    TrackBuilderChannel() {}
    TrackBuilderChannel(const edm::ParameterSet& iConfig);
    ~TrackBuilderChannel(){}
    // sets channelId of given TTTrack, return false if track outside pt range
    bool channelId(const TTTrack<Ref_Phase2TrackerDigi_>& ttTrack, int& channelId);
    // number of used channels
    int numChannels() const { return numChannels_; }
  private:
    // use tracklet seed type as channel id if False, binned track pt used if True
    bool useDuplicateRemoval_;
    // pt Boundaries in GeV, last boundary is infinity
    std::vector<double> boundaries_;
    // number of used channels
    int numChannels_;
  };

} // namespace trackFindingTracklet

EVENTSETUP_DATA_DEFAULT_RECORD(trackFindingTracklet::TrackBuilderChannel, trackFindingTracklet::TrackBuilderChannelRcd);

#endif