#include "L1Trigger/TrackFindingTracklet/interface/DuplicateRemoval.h"

#include <vector>
#include <numeric>
#include <algorithm>

namespace trklet {

  DuplicateRemoval::DuplicateRemoval(const Setup* setup, const DataFormats* dataFormats, int region, const TTDTC& ttDTC)
      : setup_(setup), dataFormats_(dataFormats), region_(region) {
    if (setup_->drUseDTCStubs()) {
      // prep dtc stub container
      int size(0);
      for (int channel : ttDTC.tfpChannels())
        for (const tt::FrameStub& frame : ttDTC.stream(region, channel))
          if (frame.first.isNonnull())
            size++;
      dtc_.reserve(size);
      // fill dtc stub container
      for (int channel : ttDTC.tfpChannels())
        for (const tt::FrameStub& frame : ttDTC.stream(region, channel))
          if (frame.first.isNonnull())
            dtc_.push_back(frame);
    }
    // unified tracklet digitisation granularity
    const double baseLr = dataFormats->base(Variable::r, Process::dr);
    const double baseLphi = dataFormats->base(Variable::phi, Process::dr);
    const double baseLz = dataFormats->base(Variable::z, Process::dr);
    // Finer granularity (by powers of 2) than the KF one. Used to transform from Tracklet to KF base.
    baseR_ = baseLr * std::pow(2, -tt::ilog2(baseLr / setup->tbBaseR()));
    basePhi_ = baseLphi * std::pow(2, -tt::ilog2(baseLphi / setup->tbBasePhi()));
    baseZ_ = baseLz * std::pow(2, -tt::ilog2(baseLz / setup->tbBaseZ()));
  }

  // read in and organize input tracks and stubs
  void DuplicateRemoval::consume(const tt::StreamsTrack& streamsTrack, const tt::StreamsStub& streamsStub) {
    auto nonNullTrack = [](int sum, const tt::FrameTrack& frame) { return sum + (frame.first.isNonnull() ? 1 : 0); };
    auto nonNullStub = [](int sum, const tt::FrameStub& frame) { return sum + (frame.first.isNonnull() ? 1 : 0); };
    // count tracks and stubs and reserve corresponding vectors
    int sizeStubs(0);
    const int offset = region_ * setup_->tmNumLayers();
    const tt::StreamTrack& streamTrack = streamsTrack[region_];
    stream_.reserve(streamTrack.size());
    const int sizeTracks = std::accumulate(streamTrack.begin(), streamTrack.end(), 0, nonNullTrack);
    for (int layer = 0; layer < setup_->tmNumLayers(); layer++) {
      const tt::StreamStub& streamStub = streamsStub[offset + layer];
      sizeStubs += std::accumulate(streamStub.begin(), streamStub.end(), 0, nonNullStub);
    }
    tracks_.reserve(sizeTracks);
    stubs_.reserve(sizeStubs);
    // store tracks and stubs
    for (int frame = 0; frame < static_cast<int>(streamTrack.size()); frame++) {
      const tt::FrameTrack& frameTrack = streamTrack[frame];
      if (frameTrack.first.isNull()) {
        stream_.push_back(nullptr);
        continue;
      }
      tracks_.emplace_back();
      Track& track = tracks_.back();
      track.stubs_ = std::vector<Stub*>(setup_->tmNumLayers(), nullptr);
      // parse track bits
      TTBV ttBV(frameTrack.second);
      track.cot_ = ttBV.val(setup_->tbWidthCot(), 0, true) * setup_->tbBaseCot();
      ttBV >>= setup_->tbWidthCot();
      track.z0_ = ttBV.val(setup_->tbWidthZ0(), 0, true) * setup_->tbBaseZ0();
      ttBV >>= setup_->tbWidthZ0();
      track.phi0_ = ttBV.val(setup_->tbWidthPhi0(), 0, true) * setup_->tbBasePhi0() - setup_->stubRangePhi() / 2.;
      ttBV >>= setup_->tbWidthPhi0();
      track.inv2R_ = -ttBV.val(setup_->tbWidthInv2R(), 0, true) * setup_->tbBaseInv2R();
      ttBV >>= setup_->tbWidthInv2R();
      const int seedType = ttBV.val(setup_->tbWidthSeedType());
      track.trackDR_ = TrackDR(frameTrack.first, dataFormats_, seedType);
      // parse stubs
      const std::vector<int>& layersSeed = setup_->tbSeedLayers(seedType);
      const std::vector<int>& layersProj = setup_->tbProjectionLayers(seedType);
      for (int layer = 0; layer < setup_->tmNumLayers(); layer++) {
        const tt::FrameStub& frameStub = streamsStub[offset + layer][frame];
        if (frameStub.first.isNull())
          continue;
        stubs_.emplace_back(frameStub.first);
        Stub& stub = stubs_.back();
        stub.sm_ = setup_->sensorModule(frameStub.first);
        const auto it = std::find(layersSeed.begin(), layersSeed.end(), stub.sm_->layerId());
        stub.layer_ = std::distance(layersSeed.begin(), it);
        if (it == layersSeed.end()) {
          stub.layer_ +=
              std::distance(layersProj.begin(), std::find(layersProj.begin(), layersProj.end(), stub.sm_->layerId()));
          const int widthR = setup_->tbWidthR(stub.sm_->type());
          const int widthPhi = setup_->tbWidthPhi();
          const int widthRZ = stub.sm_->barrel() ? setup_->tbWidthZ() : setup_->tbWidthR();
          const double baseR = setup_->stubBaseR(stub.sm_->type());
          const double basePhi =
              stub.sm_->barrel() ? setup_->tbBasePhi() : setup_->tbBasePhi(stub.sm_->layerIndexCombined());
          const double baseRZ = stub.sm_->barrel() ? setup_->tbBaseZ(stub.sm_->layerIndex()) : setup_->tbBaseR();
          TTBV ttBV(frameStub.second);
          stub.z_ = ttBV.val(widthRZ, 0, true) * baseRZ;
          ttBV >>= widthRZ;
          stub.phi_ = ttBV.val(widthPhi, 0, true) * basePhi;
          ttBV >>= widthPhi;
          stub.r_ = ttBV.val(widthR, 0, stub.sm_->barrel()) * baseR;
          ttBV >>= widthR;
          stub.stubId_ = ttBV.val(setup_->tbWidthStubId());
        } else
          stub.stubId_ = TTBV(frameStub.second).val(setup_->tbWidthStubId());
        track.stubs_[layer] = &stubs_.back();
      }
      stream_.push_back(&tracks_.back());
    }
    // remove all gaps between end and last track
    for (auto it = stream_.end(); it != stream_.begin();)
      it = (*--it) ? stream_.begin() : stream_.erase(it);
  }

  // fill output products
  void DuplicateRemoval::produce(tt::StreamsTrack& streamsTrack, tt::StreamsStub& streamsStub) {
    // remove duplicated tracks, no merge of stubs, one stub per layer expected
    algo();
    // unify stubs
    unify();
    // calc stub position and uncertainties
    pos();
    // replace stubs with DTC stubs
    if (setup_->drUseDTCStubs())
      dtc();
    // replace stubs with TT stubs
    if (setup_->drUseTTStubs())
      tt();
    // calc stub uncertainties
    delta();
    // base transformation
    redigi();
    // store output
    store(streamsTrack, streamsStub);
  }

  // remove duplicated tracks, no merge of stubs, one stub per layer expected
  void DuplicateRemoval::algo() {
    std::vector<Track*> cms(setup_->drNumComparisonModules(), nullptr);
    for (Track*& track : stream_) {
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
    for (auto it = stream_.end(); it != stream_.begin();)
      it = (*--it) ? stream_.begin() : stream_.erase(it);
  }

  // unify stubs
  void DuplicateRemoval::unify() {
    // unify stubs
    for (Track& track : tracks_) {
      for (Stub* stub : track.stubs_) {
        if (!stub)
          continue;
        if (stub->layer_ < setup_->tbNumSeedingLayers()) {
          if (stub->sm_->barrel()) {
            stub->r_ = tt::digi(setup_->stubLayerR(stub->sm_->layerIndex()), setup_->tbBaseR());
            const double z = track.z0_ + stub->r_ * track.cot_;
            if (std::abs(z) > setup_->tbMinZ() && stub->sm_->layerIndex() == 0)
              stub->r_ = tt::digi(setup_->tbInnerRadius(), setup_->tbBaseR());
          } else {
            const double z = tt::digi(setup_->stubDiskZ(stub->sm_->layerIndex()), setup_->tbBaseZ());
            const double invCot = tt::digi(1. / std::abs(track.cot_), setup_->drBaseInvCot());
            const double side = track.cot_ < 0. ? -1. : 1.;
            stub->r_ = tt::digi((z - side * track.z0_) * invCot, setup_->tbBaseR());
          }
        } else {
          if (!stub->sm_->barrel())
            stub->z_ = tt::digi(-stub->z_ * track.cot_, setup_->tbBaseZ());
          else
            stub->r_ = tt::digi(stub->r_ + tt::digi(setup_->stubLayerR(stub->sm_->layerIndex()), setup_->tbBaseR()),
                                setup_->tbBaseR());
          if (stub->sm_->type() == trackerDTC::SensorModule::Disk2S)
            stub->r_ = tt::digiR(setup_->stubDiskR(stub->sm_->layerIndex(), stub->r_), setup_->tbBaseR());
        }
      }
    }
  }

  // calc stub position
  void DuplicateRemoval::pos() {
    // unify stubs
    for (Track& track : tracks_) {
      for (Stub* stub : track.stubs_) {
        if (!stub)
          continue;
        stub->phi_ += track.phi0_ + stub->r_ * track.inv2R_;
        stub->z_ += track.z0_ + stub->r_ * track.cot_;
      }
    }
  }

  // replace stubs with DTC stubs
  void DuplicateRemoval::dtc() {
    for (Track& track : tracks_) {
      for (Stub*& stub : track.stubs_) {
        if (!stub)
          continue;
        auto via = [stub](const tt::FrameStub& fs) { return fs.first == stub->ttStubRef_; };
        const tt::FrameStub& fs = *std::find_if(dtc_.begin(), dtc_.end(), via);
        const GlobalPoint gp = setup_->stubPosDTC(fs, region_);
        stub->r_ = gp.perp();
        stub->phi_ = tt::deltaPhi(gp.phi() - region_ * setup_->regRangePhiT());
        stub->z_ = gp.z();
      }
    }
  }

  // replace stubs with TT stubs
  void DuplicateRemoval::tt() {
    for (Track& track : tracks_) {
      for (Stub*& stub : track.stubs_) {
        if (!stub)
          continue;
        const GlobalPoint gp = setup_->stubPosTT(stub->ttStubRef_);
        stub->r_ = gp.perp();
        stub->phi_ = tt::deltaPhi(gp.phi() - region_ * setup_->regRangePhiT());
        stub->z_ = gp.z();
      }
    }
  }

  // calc stub uncertainties
  void DuplicateRemoval::delta() {
    // unify stubs
    for (Track& track : tracks_) {
      for (Stub* stub : track.stubs_) {
        if (!stub)
          continue;
        stub->dPhi_ = stub->sm_->dPhi(stub->r_, track.inv2R_) / 2.;
        stub->dZ_ = stub->sm_->dZ(track.cot_) / 2.;
      }
    }
  }

  // base transformation
  void DuplicateRemoval::redigi() {
    for (Track* track : stream_) {
      if (!track)
        continue;
      TTBV hitPattern(0, setup_->drNumLayers());
      for (Stub* stub : track->stubs_) {
        if (!stub)
          continue;
        stub->r_ = tt::digi(stub->r_, setup_->tbBaseR());
        stub->phi_ = tt::digi(stub->phi_, setup_->tbBasePhi());
        stub->z_ = tt::digi(stub->z_, setup_->tbBaseZ());
        stub->dPhi_ = tt::digi(stub->dPhi_, setup_->tbBasePhi());
        stub->dZ_ = tt::digi(stub->dZ_, setup_->tbBaseZ());
        // base transform into high precision format
        stub->r_ = tt::redigi(stub->r_, setup_->tbBaseR(), baseR_, setup_->widthDSPbu());
        stub->phi_ = tt::redigi(stub->phi_, setup_->tbBasePhi(), basePhi_, setup_->widthDSPbu());
        stub->z_ = tt::redigi(stub->z_, setup_->tbBaseZ(), baseZ_, setup_->widthDSPbu());
        stub->dPhi_ = tt::redigi(stub->dPhi_, setup_->tbBasePhi(), basePhi_, setup_->widthDSPbu());
        stub->dZ_ = tt::redigi(stub->dZ_, setup_->tbBaseZ(), baseZ_, setup_->widthDSPbu());
      }
    }
  }

  // store output
  void DuplicateRemoval::store(tt::StreamsTrack& streamsTrack, tt::StreamsStub& streamsStub) const {
    const int offset = region_ * setup_->drNumLayers();
    tt::StreamTrack& streamTrack = streamsTrack[region_];
    streamTrack.reserve(stream_.size());
    for (int layer = 0; layer < setup_->drNumLayers(); layer++)
      streamsStub[offset + layer].reserve(stream_.size());
    for (Track* track : stream_) {
      if (!track) {
        streamTrack.emplace_back();
        for (int layer = 0; layer < setup_->drNumLayers(); layer++)
          streamsStub[offset + layer].emplace_back();
        continue;
      }
      streamTrack.push_back(track->trackDR_.frame());
      TTBV hitPattern(0, setup_->drNumLayers());
      for (Stub* stub : track->stubs_) {
        if (!stub)
          continue;
        hitPattern.set(stub->layer_);
        streamsStub[offset + stub->layer_].emplace_back(
            StubDR(stub->ttStubRef_, dataFormats_, stub->r_, stub->phi_, stub->z_, stub->dPhi_, stub->dZ_).frame());
      }
      // fill gaps
      for (int id : hitPattern.ids(false))
        streamsStub[offset + id].emplace_back();
    }
  }

  // compares two tracks, returns true if those are considered duplicates
  bool DuplicateRemoval::equalEnough(Track* t0, Track* t1) const {
    int same(0);
    for (int layer = 0; layer < setup_->tmNumLayers(); layer++) {
      const Stub* s0 = t0->stubs_[layer];
      const Stub* s1 = t1->stubs_[layer];
      if (s0 && s1 && s0->stubId_ == s1->stubId_)
        same++;
    }
    return same >= setup_->drMinIdenticalStubs();
  }

}  // namespace trklet
