#include "L1Trigger/TrackFindingTracklet/interface/DuplicateRemoval.h"

#include <vector>
#include <numeric>
#include <algorithm>

namespace trklet {

  DuplicateRemoval::DuplicateRemoval(const tt::Setup* setup,
                                     const trackerTFP::LayerEncoding* layerEncoding,
                                     const DataFormats* dataFormats,
                                     const ChannelAssignment* channelAssignment,
                                     int region)
      : setup_(setup),
        layerEncoding_(layerEncoding),
        dataFormats_(dataFormats),
        channelAssignment_(channelAssignment),
        region_(region) {
    const DataFormat& r = dataFormats_->format(Variable::r, Process::dr);
    const int width = setup_->widthAddrBRAM18() - 1;
    const double base = r.base() * pow(2., r.width() - width);
    const double range = r.range();
    r_ = DataFormat(true, width, base, range);
    tmNumLayers_ = channelAssignment_->tmNumLayers();
    phi_ = dataFormats_->format(Variable::phi, Process::dr);
  }

  // read in and organize input tracks and stubs
  void DuplicateRemoval::consume(const tt::StreamsTrack& streamsTrack, const tt::StreamsStub& streamsStub) {
    auto nonNullTrack = [](int sum, const tt::FrameTrack& frame) { return sum + (frame.first.isNonnull() ? 1 : 0); };
    auto nonNullStub = [](int sum, const tt::FrameStub& frame) { return sum + (frame.first.isNonnull() ? 1 : 0); };
    // count tracks and stubs and reserve corresponding vectors
    int sizeStubs(0);
    const int offset = region_ * tmNumLayers_;
    const tt::StreamTrack& streamTrack = streamsTrack[region_];
    input_.reserve(streamTrack.size());
    const int sizeTracks = std::accumulate(streamTrack.begin(), streamTrack.end(), 0, nonNullTrack);
    for (int layer = 0; layer < tmNumLayers_; layer++) {
      const tt::StreamStub& streamStub = streamsStub[offset + layer];
      sizeStubs += std::accumulate(streamStub.begin(), streamStub.end(), 0, nonNullStub);
    }
    tracks_.reserve(sizeTracks);
    stubs_.reserve(sizeStubs);
    // transform input data into handy structs
    for (int frame = 0; frame < static_cast<int>(streamTrack.size()); frame++) {
      const tt::FrameTrack& frameTrack = streamTrack[frame];
      if (frameTrack.first.isNull()) {
        input_.push_back(nullptr);
        continue;
      }
      // lookup layerEncoding
      const TrackTM track(frameTrack, dataFormats_);
      const double inv2R = abs(track.inv2R());
      const double zT = abs(track.zT());
      const double cot = zT / setup_->chosenRofZ();
      const std::vector<int>& layerEncoding = layerEncoding_->layerEncoding(zT);
      std::vector<Stub*> stubs(tmNumLayers_, nullptr);
      TTBV hitPattern(0, setup_->numLayers());
      for (int layer = 0; layer < tmNumLayers_; layer++) {
        const tt::FrameStub& frameStub = streamsStub[offset + layer][frame];
        const TTStubRef& ttStubRef = frameStub.first;
        if (ttStubRef.isNull())
          continue;
        // encode layerId
        const int decodedLayerId =
            layer + setup_->offsetLayerId() +
            (layer < setup_->numBarrelLayer() ? 0 : setup_->offsetLayerDisks() - setup_->numBarrelLayer());
        const auto it = std::find(layerEncoding.begin(), layerEncoding.end(), decodedLayerId);
        const int encodedLayerId =
            std::min(static_cast<int>(std::distance(layerEncoding.begin(), it)), setup_->numLayers() - 1);
        // kill stub on already occupied layer
        if (hitPattern.test(encodedLayerId))
          continue;
        hitPattern.set(encodedLayerId);
        const StubTM stubTM(frameStub, dataFormats_);
        const int stubId = stubTM.stubId() / 2;
        const bool psTilt = stubTM.stubId() % 2 == 1;
        // calculate stub uncertainties
        static constexpr int numBarrelPSLayer = 3;
        const bool barrel = layer < setup_->numBarrelLayer();
        const bool ps = barrel ? layer < numBarrelPSLayer : psTilt;
        const bool tilt = barrel && psTilt;
        const double length = .5 * (ps ? setup_->pitchColPS() : setup_->pitchCol2S());
        const double pitch = .5 * (ps ? setup_->pitchRowPS() : setup_->pitchRow2S());
        const double pitchOverR = phi_.digi(pitch / (r_.digi(stubTM.r()) + setup_->chosenRofPhi()));
        double lengthZ = length;
        double lengthR = 0.;
        if (!barrel) {
          lengthZ = length * cot;
          lengthR = length;
        } else if (tilt) {
          lengthZ = length * (setup_->tiltApproxSlope() * cot + setup_->tiltApproxIntercept());
          lengthR = .5 * setup_->tiltUncertaintyR();
        }
        const double dR = lengthR + .5 * setup_->scattering();
        const double dZ = lengthZ;
        const double dPhi = phi_.digi(dR * inv2R) + pitchOverR;
        const StubDR stubDR(stubTM, stubTM.r(), stubTM.phi(), stubTM.z(), dPhi, dZ);
        stubs_.emplace_back(stubDR.frame(), stubId, encodedLayerId);
        stubs[layer] = &stubs_.back();
      }
      // kill tracks with not enough layers
      if (hitPattern.count() < setup_->kfMinLayers()) {
        input_.push_back(nullptr);
        continue;
      }
      tracks_.emplace_back(frameTrack, stubs);
      input_.push_back(&tracks_.back());
    }
    // remove all gaps between end and last track
    for (auto it = input_.end(); it != input_.begin();)
      it = (*--it) ? input_.begin() : input_.erase(it);
  }

  // fill output products
  void DuplicateRemoval::produce(tt::StreamsTrack& streamsTrack, tt::StreamsStub& streamsStub) {
    const int offset = region_ * setup_->numLayers();
    // remove duplicated tracks, no merge of stubs, one stub per layer expected
    std::vector<Track*> cms(channelAssignment_->numComparisonModules(), nullptr);
    for (Track*& track : input_) {
      if (!track)
        // gaps propagate through chain and appear in output stream
        continue;
      for (Track*& trackCM : cms) {
        if (!trackCM) {
          // tracks used in CMs propagate through chain and do appear in output stream unaltered
          trackCM = track;
          break;
        }
        if (equalEnough(track, trackCM)) {
          // tracks compared in CMs propagate through chain and appear in output stream as gap if identified as duplicate or unaltered elsewise
          track = nullptr;
          break;
        }
      }
    }
    // remove all gaps between end and last track
    for (auto it = input_.end(); it != input_.begin();)
      it = (*--it) ? input_.begin() : input_.erase(it);
    // store output
    tt::StreamTrack& streamTrack = streamsTrack[region_];
    streamTrack.reserve(input_.size());
    for (int layer = 0; layer < setup_->numLayers(); layer++)
      streamsStub[offset + layer].reserve(input_.size());
    for (Track* track : input_) {
      if (!track) {
        streamTrack.emplace_back(tt::FrameTrack());
        for (int layer = 0; layer < setup_->numLayers(); layer++)
          streamsStub[offset + layer].emplace_back(tt::FrameStub());
        continue;
      }
      streamTrack.push_back(track->frame_);
      TTBV hitPattern(0, setup_->numLayers());
      for (Stub* stub : track->stubs_) {
        if (!stub)
          continue;
        hitPattern.set(stub->layer_);
        streamsStub[offset + stub->layer_].emplace_back(stub->frame_);
      }
      for (int layer : hitPattern.ids(false))
        streamsStub[offset + layer].emplace_back(tt::FrameStub());
    }
  }

  // compares two tracks, returns true if those are considered duplicates
  bool DuplicateRemoval::equalEnough(Track* t0, Track* t1) const {
    int same(0);
    for (int layer = 0; layer < channelAssignment_->tmNumLayers(); layer++) {
      Stub* s0 = t0->stubs_[layer];
      Stub* s1 = t1->stubs_[layer];
      if (s0 && s1 && s0->stubId_ == s1->stubId_)
        same++;
    }
    return same >= channelAssignment_->minIdenticalStubs();
  }

}  // namespace trklet
