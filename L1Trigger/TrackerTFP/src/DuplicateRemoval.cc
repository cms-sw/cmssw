#include "L1Trigger/TrackerTFP/interface/DuplicateRemoval.h"

#include <numeric>
#include <algorithm>
#include <iterator>
#include <deque>
#include <vector>
#include <set>
#include <utility>
#include <cmath>

namespace trackerTFP {

  DuplicateRemoval::DuplicateRemoval(const tt::Setup* setup,
                                     const DataFormats* dataFormats,
                                     std::vector<TrackDR>& tracks,
                                     std::vector<StubDR>& stubs)
      : setup_(setup), dataFormats_(dataFormats), tracks_(tracks), stubs_(stubs) {
    numChannel_ = dataFormats_->numChannel(Process::kf);
    numLayers_ = setup_->numLayers();
    numInv2R_ = setup_->htNumBinsInv2R() + 2;
    numPhiT_ = setup_->htNumBinsPhiT() * setup_->gpNumBinsPhiT();
    numZT_ = setup_->gpNumBinsZT();
  }

  // fill output products
  void DuplicateRemoval::produce(const std::vector<std::vector<TrackKF*>>& tracksIn,
                                 const std::vector<std::vector<StubKF*>>& stubsIn,
                                 std::vector<std::vector<TrackDR*>>& tracksOut,
                                 std::vector<std::vector<StubDR*>>& stubsOut) {
    int nTracks(0);
    for (const std::vector<TrackKF*>& tracks : tracksIn)
      nTracks += std::accumulate(
          tracks.begin(), tracks.end(), 0, [](int sum, TrackKF* track) { return sum + (track ? 1 : 0); });
    std::vector<Track> tracks;
    tracks.reserve(nTracks);
    std::deque<Track*> stream;
    // merge 4 channel to 1
    for (int channel = numChannel_ - 1; channel >= 0; channel--) {
      const int offset = channel * numLayers_;
      const std::vector<TrackKF*>& tracksChannel = tracksIn[channel];
      for (int frame = 0; frame < static_cast<int>(tracksChannel.size()); frame++) {
        TrackKF* track = tracksChannel[frame];
        if (!track) {
          stream.push_back(nullptr);
          continue;
        }
        std::vector<StubKF*> stubs;
        stubs.reserve(numLayers_);
        for (int layer = 0; layer < numLayers_; layer++)
          stubs.push_back(stubsIn[offset + layer][frame]);
        const bool match = track->match().val();
        const int inv2R = dataFormats_->format(Variable::inv2R, Process::ht).integer(track->inv2R()) + numInv2R_ / 2;
        const int phiT = dataFormats_->format(Variable::phiT, Process::ht).integer(track->phiT()) + numPhiT_ / 2;
        const int zT = dataFormats_->format(Variable::zT, Process::gp).integer(track->zT()) + numZT_ / 2;
        tracks.emplace_back(track, stubs, match, inv2R, phiT, zT);
        stream.push_back(&tracks.back());
      }
    }
    // truncate if desired
    if (setup_->enableTruncation() && static_cast<int>(stream.size()) > setup_->numFramesHigh()) {
      const auto limit = std::next(stream.begin(), setup_->numFramesHigh());
      stream.erase(limit, stream.end());
    }
    // remove duplicates
    std::vector<Track*> killed;
    killed.reserve(stream.size());
    std::vector<std::vector<TTBV>> hits(numZT_, std::vector<TTBV>(numInv2R_, TTBV(0, numPhiT_)));
    for (Track*& track : stream) {
      if (!track)
        continue;
      if (track->match_) {
        hits[track->zT_][track->inv2R_].set(track->phiT_);
      } else {
        killed.push_back(track);
        track = nullptr;
      }
    }
    // restore duplicates
    for (Track* track : killed) {
      if (hits[track->zT_][track->inv2R_][track->phiT_]) {
        stream.push_back(nullptr);
        continue;
      }
      hits[track->zT_][track->inv2R_].set(track->phiT_);
      stream.push_back(track);
    }
    // truncate
    if (setup_->enableTruncation() && static_cast<int>(stream.size()) > setup_->numFramesHigh()) {
      const auto limit = std::next(stream.begin(), setup_->numFramesHigh());
      stream.erase(limit, stream.end());
    }
    // remove trailing nullptr
    for (auto it = stream.end(); it != stream.begin();)
      it = (*--it) ? stream.begin() : stream.erase(it);
    // convert and store tracks
    tracksOut[0].reserve(stream.size());
    for (std::vector<StubDR*>& layer : stubsOut)
      layer.reserve(stream.size());
    for (Track* track : stream) {
      if (!track) {
        tracksOut[0].push_back(nullptr);
        for (std::vector<StubDR*>& layer : stubsOut)
          layer.push_back(nullptr);
        continue;
      }
      const DataFormat& gp = dataFormats_->format(Variable::zT, Process::gp);
      TrackKF* trackKF = track->track_;
      const double inv2R = trackKF->inv2R();
      const double phiT = trackKF->phiT();
      const double zT = trackKF->zT();
      const double cot = trackKF->cot() + gp.digi(zT) / setup_->chosenRofZ();
      tracks_.emplace_back(*trackKF, inv2R, phiT, cot, zT);
      tracksOut[0].push_back(&tracks_.back());
      for (int layer = 0; layer < numLayers_; layer++) {
        std::vector<StubDR*>& layerStubs = stubsOut[layer];
        StubKF* stub = track->stubs_[layer];
        if (!stub) {
          layerStubs.push_back(nullptr);
          continue;
        }
        const double r = stub->r();
        const double phi = stub->phi();
        const double z = stub->z();
        const double dPhi = stub->dPhi();
        const double dZ = stub->dZ();
        stubs_.emplace_back(*stub, r, phi, z, dPhi, dZ);
        layerStubs.push_back(&stubs_.back());
      }
    }
  }

}  // namespace trackerTFP
