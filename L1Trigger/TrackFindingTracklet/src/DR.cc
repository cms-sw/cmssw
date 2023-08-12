#include "L1Trigger/TrackFindingTracklet/interface/DR.h"

#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;
using namespace edm;
using namespace tt;
using namespace trackerTFP;

namespace trklet {

  DR::DR(const ParameterSet& iConfig,
         const Setup* setup,
         const DataFormats* dataFormats,
         const ChannelAssignment* channelAssignment,
         int region)
      : enableTruncation_(iConfig.getParameter<bool>("EnableTruncation")),
        setup_(setup),
        dataFormats_(dataFormats),
        channelAssignment_(channelAssignment),
        region_(region),
        input_(channelAssignment_->numNodesDR()) {}

  // read in and organize input tracks and stubs
  void DR::consume(const StreamsTrack& streamsTrack, const StreamsStub& streamsStub) {
    const int offsetTrack = region_ * channelAssignment_->numNodesDR();
    auto nonNullTrack = [](int& sum, const FrameTrack& frame) { return sum += (frame.first.isNonnull() ? 1 : 0); };
    auto nonNullStub = [](int& sum, const FrameStub& frame) { return sum += (frame.first.isNonnull() ? 1 : 0); };
    // count tracks and stubs and reserve corresponding vectors
    int sizeTracks(0);
    int sizeStubs(0);
    for (int channel = 0; channel < channelAssignment_->numNodesDR(); channel++) {
      const int streamTrackId = offsetTrack + channel;
      const int offsetStub = streamTrackId * setup_->numLayers();
      const StreamTrack& streamTrack = streamsTrack[streamTrackId];
      input_[channel].reserve(streamTrack.size());
      sizeTracks += accumulate(streamTrack.begin(), streamTrack.end(), 0, nonNullTrack);
      for (int layer = 0; layer < setup_->numLayers(); layer++) {
        const StreamStub& streamStub = streamsStub[offsetStub + layer];
        sizeStubs += accumulate(streamStub.begin(), streamStub.end(), 0, nonNullStub);
      }
    }
    tracks_.reserve(sizeTracks);
    stubs_.reserve(sizeStubs);
    // transform input data into handy structs
    for (int channel = 0; channel < channelAssignment_->numNodesDR(); channel++) {
      vector<Track*>& input = input_[channel];
      const int streamTrackId = offsetTrack + channel;
      const int offsetStub = streamTrackId * setup_->numLayers();
      const StreamTrack& streamTrack = streamsTrack[streamTrackId];
      for (int frame = 0; frame < (int)streamTrack.size(); frame++) {
        const FrameTrack& frameTrack = streamTrack[frame];
        if (frameTrack.first.isNull()) {
          input.push_back(nullptr);
          continue;
        }
        vector<Stub*> stubs;
        stubs.reserve(setup_->numLayers());
        for (int layer = 0; layer < setup_->numLayers(); layer++) {
          const FrameStub& frameStub = streamsStub[offsetStub + layer][frame];
          if (frameStub.first.isNull())
            continue;
          TTBV ttBV = frameStub.second;
          const TTBV z(ttBV, dataFormats_->format(Variable::z, Process::kfin).width(), 0, true);
          ttBV >>= dataFormats_->format(Variable::z, Process::kfin).width();
          const TTBV phi(ttBV, dataFormats_->format(Variable::phi, Process::kfin).width(), 0, true);
          ttBV >>= dataFormats_->format(Variable::phi, Process::kfin).width();
          const TTBV r(ttBV, dataFormats_->format(Variable::r, Process::kfin).width(), 0, true);
          ttBV >>= dataFormats_->format(Variable::r, Process::kfin).width();
          const TTBV stubId(ttBV, channelAssignment_->widthSeedStubId(), 0);
          ttBV >>= channelAssignment_->widthSeedStubId();
          const TTBV layerId(ttBV, channelAssignment_->widthLayerId(), 0);
          ttBV >>= channelAssignment_->widthLayerId();
          const TTBV tilt(ttBV, channelAssignment_->widthPSTilt(), 0);
          const FrameStub frame(frameStub.first,
                                Frame("1" + tilt.str() + layerId.str() + r.str() + phi.str() + z.str()));
          stubs_.emplace_back(frame, stubId.val(), layer);
          stubs.push_back(&stubs_.back());
        }
        tracks_.emplace_back(frameTrack, stubs);
        input.push_back(&tracks_.back());
      }
      // remove all gaps between end and last track
      for (auto it = input.end(); it != input.begin();)
        it = (*--it) ? input.begin() : input.erase(it);
    }
  }

  // fill output products
  void DR::produce(StreamsStub& accpetedStubs,
                   StreamsTrack& acceptedTracks,
                   StreamsStub& lostStubs,
                   StreamsTrack& lostTracks) {
    const int offsetTrack = region_ * channelAssignment_->numNodesDR();
    for (int node = 0; node < channelAssignment_->numNodesDR(); node++) {
      const int channelTrack = offsetTrack + node;
      const int offsetStub = channelTrack * setup_->numLayers();
      // remove duplicated tracks, no merge of stubs, one stub per layer expected
      vector<Track*> cms(channelAssignment_->numComparisonModules(), nullptr);
      vector<Track*>& tracks = input_[node];
      for (Track*& track : tracks) {
        if (!track)
          // gaps propagate trough chain and appear in output stream
          continue;
        for (Track*& trackCM : cms) {
          if (!trackCM) {
            // tracks used in CMs propagate trough chain and appear in output stream unaltered
            trackCM = track;
            break;
          }
          if (equalEnough(track, trackCM)) {
            // tracks compared in CMs propagate trough chain and appear in output stream as gap if identified as duplicate or unaltered elsewise
            track = nullptr;
            break;
          }
        }
      }
      // remove all gaps between end and last track
      for (auto it = tracks.end(); it != tracks.begin();)
        it = (*--it) ? tracks.begin() : tracks.erase(it);
      // store output
      StreamTrack& streamTrack = acceptedTracks[channelTrack];
      streamTrack.reserve(tracks.size());
      for (int layer = 0; layer < setup_->numLayers(); layer++)
        accpetedStubs[offsetStub + layer].reserve(tracks.size());
      for (Track* track : tracks) {
        if (!track) {
          streamTrack.emplace_back(FrameTrack());
          for (int layer = 0; layer < setup_->numLayers(); layer++)
            accpetedStubs[offsetStub + layer].emplace_back(FrameStub());
          continue;
        }
        streamTrack.push_back(track->frame_);
        TTBV hitPattern(0, setup_->numLayers());
        for (Stub* stub : track->stubs_) {
          hitPattern.set(stub->channel_);
          accpetedStubs[offsetStub + stub->channel_].push_back(stub->frame_);
        }
        for (int layer : hitPattern.ids(false))
          accpetedStubs[offsetStub + layer].emplace_back(FrameStub());
      }
    }
  }

  // compares two tracks, returns true if those are considered duplicates
  bool DR::equalEnough(Track* t0, Track* t1) const {
    int same(0);
    for (int layer = 0; layer < setup_->numLayers(); layer++) {
      auto onLayer = [layer](Stub* stub) { return stub->channel_ == layer; };
      const auto s0 = find_if(t0->stubs_.begin(), t0->stubs_.end(), onLayer);
      const auto s1 = find_if(t1->stubs_.begin(), t1->stubs_.end(), onLayer);
      if (s0 != t0->stubs_.end() && s1 != t1->stubs_.end() && **s0 == **s1)
        same++;
    }
    return same >= channelAssignment_->minIdenticalStubs();
  }

}  // namespace trklet