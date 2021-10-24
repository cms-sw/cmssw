#include "L1Trigger/TrackFindingTracklet/interface/TrackBuilderChannel.h"

#include <vector>

using namespace std;
using namespace edm;

namespace trackFindingTracklet {

  TrackBuilderChannel::TrackBuilderChannel(const edm::ParameterSet& iConfig) :
    useDuplicateRemoval_(iConfig.getParameter<bool>("UseDuplicateRemoval")),
    boundaries_(iConfig.getParameter<vector<double>>("PtBoundaries")),
    numChannels_(useDuplicateRemoval_ ? 2 * boundaries_.size() : iConfig.getParameter<int>("NumSeedTypes"))
  {}

  // sets channelId of given TTTrack, return false if track outside pt range
  bool TrackBuilderChannel::channelId(const TTTrack<Ref_Phase2TrackerDigi_>& ttTrack, int& channelId) {
    if (!useDuplicateRemoval_)
      return ttTrack.trackSeedType();
    const double pt = ttTrack.momentum().perp();
    channelId = -1;
    for (double boundary : boundaries_) {
      if (pt < boundary)
        break;
      else
        channelId++;
    }
    if (channelId == -1)
      return false;
    channelId = ttTrack.rInv() < 0. ? channelId : numChannels_ - channelId - 1;
    channelId += ttTrack.phiSector() * numChannels_;
    return true;
  }

} // namespace trackFindingTracklet