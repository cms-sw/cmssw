#include "L1Trigger/TrackFindingTracklet/interface/TrackMultiplexer.h"

#include <vector>
#include <deque>
#include <set>
#include <numeric>
#include <algorithm>

namespace trklet {

  TrackMultiplexer::TrackMultiplexer(const Setup* setup, int region)
      : setup_(setup), region_(region), input_(setup_->tbNumSeedTypes()) {
    streamsTrack_.reserve(setup_->tbNumSeedTypes());
    streamsStub_.reserve(setup_->tbNumSeedTypes() * setup_->tbNumLayers());
  }

  // read in and organize input tracks and stubs
  void TrackMultiplexer::consume(const tt::StreamsTrack& streamsTrack, const tt::StreamsStub& streamsStub) {
    const int offsetTrack = region_ * setup_->tbNumSeedTypes();
    // prep input container
    int nTracks(0);
    for (int seedType = 0; seedType < setup_->tbNumSeedTypes(); seedType++) {
      const int channelTrack = offsetTrack + seedType;
      const int offsetStub = channelTrack * setup_->tbNumLayers();
      const tt::StreamTrack& stream = streamsTrack[channelTrack];
      input_[seedType].reserve(stream.size());
      nTracks += std::accumulate(stream.begin(), stream.end(), 0, [](int sum, const tt::FrameTrack f) {
        return sum += (f.first.isNull() ? 0 : 1);
      });
      streamsTrack_.push_back(stream);
      for (int layer = 0; layer < setup_->tbNumLayers(); layer++)
        streamsStub_.push_back(streamsStub[offsetStub + layer]);
    }
    tracks_.reserve(nTracks);
    // store tracks
    for (int channel = 0; channel < setup_->tbNumSeedTypes(); channel++) {
      const int channelTrack = offsetTrack + channel;
      std::vector<Track*>& input = input_[channel];
      const tt::StreamTrack streamTrack = streamsTrack[channelTrack];
      for (int frame = 0; frame < static_cast<int>(streamTrack.size()); frame++) {
        const tt::FrameTrack frameTrack = streamTrack[frame];
        Track* track(nullptr);
        if (frameTrack.first.isNonnull()) {
          tracks_.emplace_back(channel, frame);
          track = &tracks_.back();
        }
        input.push_back(track);
      }
    }
  }

  // fill output products
  void TrackMultiplexer::produce(tt::StreamsTrack& streamsTrack, tt::StreamsStub& streamsStub) {
    // emualte clock domain crossing
    static constexpr int ticksPerGap = 3;
    static constexpr int gapPos = 1;
    std::vector<std::deque<Track*>> streams(setup_->tbNumSeedTypes());
    for (int seedType = 0; seedType < setup_->tbNumSeedTypes(); seedType++) {
      int iTrack(0);
      std::deque<Track*>& stream = streams[seedType];
      const std::vector<Track*>& intput = input_[seedType];
      for (int tick = 0; iTrack < (int)intput.size(); tick++)
        stream.push_back(tick % ticksPerGap != gapPos ? intput[iTrack++] : nullptr);
    }
    // remove all gaps between end and last track
    for (std::deque<Track*>& stream : streams)
      for (auto it = stream.end(); it != stream.begin();)
        it = (*--it) ? stream.begin() : stream.erase(it);
    // route into single channel
    std::deque<Track*> accepted;
    std::vector<std::deque<Track*>> stacks(setup_->tbNumSeedTypes());
    // clock accurate firmware emulation, each while trip describes one clock tick, one stub in and one stub out per tick
    auto empty = [](const std::deque<Track*>& tracks) { return tracks.empty(); };
    while (!std::all_of(streams.begin(), streams.end(), empty) || !std::all_of(stacks.begin(), stacks.end(), empty)) {
      // fill input fifos
      for (int seedType = 0; seedType < setup_->tbNumSeedTypes(); seedType++) {
        Track* track = pop_front(streams[seedType]);
        if (track)
          stacks[seedType].push_back(track);
      }
      // merge input fifos to one stream, prioritizing lower input channel over higher channel, affects DR
      bool nothingToRoute(true);
      for (int seedType : setup_->tmMuxOrder()) {
        Track* track = pop_front(stacks[seedType]);
        if (track) {
          nothingToRoute = false;
          accepted.push_back(track);
          break;
        }
      }
      if (nothingToRoute)
        accepted.push_back(nullptr);
    }
    // truncate if desired
    if (setup_->enableTruncation() && static_cast<int>(accepted.size()) > setup_->numFrames())
      accepted.resize(setup_->numFrames());
    // remove all gaps between end and last track
    for (auto it = accepted.end(); it != accepted.begin();)
      it = (*--it) ? accepted.begin() : accepted.erase(it);
    // prep output container
    const int offsetOut = region_ * setup_->tmNumLayers();
    tt::StreamTrack& streamTrack = streamsTrack[region_];
    streamTrack.reserve(accepted.size());
    for (int layer = 0; layer < setup_->tmNumLayers(); layer++)
      streamsStub[offsetOut + layer].reserve(accepted.size());
    // fill output tracks and stubs
    for (Track* track : accepted) {
      if (!track) {  // fill gaps
        streamTrack.emplace_back(tt::FrameTrack());
        for (int layer = 0; layer < setup_->tmNumLayers(); layer++)
          streamsStub[offsetOut + layer].emplace_back(tt::FrameStub());
        continue;
      }
      streamTrack.emplace_back(streamsTrack_[track->channel_][track->frame_]);
      const int offsetIn = track->channel_ * setup_->tbNumLayers();
      for (int layer = 0; layer < setup_->tbNumLayers(); layer++)
        streamsStub[offsetOut + layer].emplace_back(streamsStub_[offsetIn + layer][track->frame_]);
    }
  }

  // remove and return first element of deque, returns nullptr if empty
  template <class T>
  T* TrackMultiplexer::pop_front(std::deque<T*>& ts) const {
    T* t = nullptr;
    if (!ts.empty()) {
      t = ts.front();
      ts.pop_front();
    }
    return t;
  }

}  // namespace trklet
