#include "L1Trigger/TrackFindingTracklet/interface/TrackMultiplexer.h"

#include <vector>
#include <deque>
#include <set>
#include <numeric>
#include <algorithm>

namespace trklet {

  TrackMultiplexer::TrackMultiplexer(const tt::Setup* setup,
                                     const DataFormats* dataFormats,
                                     const ChannelAssignment* channelAssignment,
                                     const Settings* settings,
                                     int region)
      : setup_(setup),
        dataFormats_(dataFormats),
        channelAssignment_(channelAssignment),
        settings_(settings),
        region_(region),
        input_(channelAssignment_->numChannelsTrack()) {
    // unified tracklet digitisation granularity
    baseUinv2R_ = .5 * settings_->kphi1() / settings_->kr() * pow(2, settings_->rinv_shift());
    baseUphiT_ = settings_->kphi1() * pow(2, settings_->phi0_shift());
    baseUcot_ = settings_->kz() / settings_->kr() * pow(2, settings_->t_shift());
    baseUzT_ = settings_->kz() * pow(2, settings_->z0_shift());
    baseUr_ = settings_->kr();
    baseUphi_ = settings_->kphi1();
    baseUz_ = settings_->kz();
    // DR input format digitisation granularity (identical to TMTT)
    baseLinv2R_ = dataFormats->base(Variable::inv2R, Process::tm);
    baseLphiT_ = dataFormats->base(Variable::phiT, Process::tm);
    baseLzT_ = dataFormats->base(Variable::zT, Process::tm);
    baseLr_ = dataFormats->base(Variable::r, Process::tm);
    baseLphi_ = dataFormats->base(Variable::phi, Process::tm);
    baseLz_ = dataFormats->base(Variable::z, Process::tm);
    baseLcot_ = baseLz_ / baseLr_;
    // Finer granularity (by powers of 2) than the TMTT one. Used to transform from Tracklet to TMTT base.
    baseHinv2R_ = baseLinv2R_ * pow(2, floor(log2(baseUinv2R_ / baseLinv2R_)));
    baseHphiT_ = baseLphiT_ * pow(2, floor(log2(baseUphiT_ / baseLphiT_)));
    baseHzT_ = baseLzT_ * pow(2, floor(log2(baseUzT_ / baseLzT_)));
    baseHr_ = baseLr_ * pow(2, floor(log2(baseUr_ / baseLr_)));
    baseHphi_ = baseLphi_ * pow(2, floor(log2(baseUphi_ / baseLphi_)));
    baseHz_ = baseLz_ * pow(2, floor(log2(baseUz_ / baseLz_)));
    baseHcot_ = baseLcot_ * pow(2, floor(log2(baseUcot_ / baseLcot_)));
    // calculate digitisation granularity used for inverted cot(theta)
    const int baseShiftInvCot = ceil(log2(setup_->outerRadius() / setup_->hybridRangeR())) - setup_->widthDSPbu();
    baseInvCot_ = pow(2, baseShiftInvCot);
    const int unusedMSBScot =
        floor(log2(baseUcot_ * pow(2.0, channelAssignment_->tmWidthCot()) / 2. / setup_->maxCot()));
    const int baseShiftScot = channelAssignment_->tmWidthCot() - unusedMSBScot - 1 - setup_->widthAddrBRAM18();
    baseScot_ = baseUcot_ * pow(2.0, baseShiftScot);
  }

  // read in and organize input tracks and stubs
  void TrackMultiplexer::consume(const tt::StreamsTrack& streamsTrack, const tt::StreamsStub& streamsStub) {
    const int offsetTrack = region_ * channelAssignment_->numChannelsTrack();
    // count tracks and stubs to reserve container
    int nTracks(0);
    int nStubs(0);
    for (int channel = 0; channel < channelAssignment_->numChannelsTrack(); channel++) {
      const int channelTrack = offsetTrack + channel;
      const int offsetStub = channelAssignment_->offsetStub(channelTrack);
      const int numProjectionLayers = channelAssignment_->numProjectionLayers(channel);
      const tt::StreamTrack& streamTrack = streamsTrack[channelTrack];
      input_[channel].reserve(streamTrack.size());
      for (int frame = 0; frame < static_cast<int>(streamTrack.size()); frame++) {
        if (streamTrack[frame].first.isNull())
          continue;
        nTracks++;
        for (int layer = 0; layer < numProjectionLayers; layer++)
          if (streamsStub[offsetStub + layer][frame].first.isNonnull())
            nStubs++;
      }
    }
    stubs_.reserve(nStubs + nTracks * channelAssignment_->numSeedingLayers());
    tracks_.reserve(nTracks);
    // store tracks and stubs
    for (int channel = 0; channel < channelAssignment_->numChannelsTrack(); channel++) {
      const int numP = channelAssignment_->numProjectionLayers(channel);
      const int channelTrack = offsetTrack + channel;
      const int offsetStub = channelAssignment_->offsetStub(channelTrack);
      const tt::StreamTrack& streamTrack = streamsTrack[channelTrack];
      std::vector<Track*>& input = input_[channel];
      for (int frame = 0; frame < static_cast<int>(streamTrack.size()); frame++) {
        const TTTrackRef& ttTrackRef = streamTrack[frame].first;
        if (ttTrackRef.isNull()) {
          input.push_back(nullptr);
          continue;
        }
        //convert track parameter
        const double offset = region_ * setup_->baseRegion();
        double inv2R = digi(-ttTrackRef->rInv() / 2., baseUinv2R_);
        const double phi0U = digi(tt::deltaPhi(ttTrackRef->phi() - offset + setup_->hybridRangePhi() / 2.), baseUphiT_);
        const double phi0S = digi(phi0U - setup_->hybridRangePhi() / 2., baseUphiT_);
        double cot = digi(ttTrackRef->tanL(), baseUcot_);
        double z0 = digi(ttTrackRef->z0(), baseUzT_);
        double phiT = digi(phi0S + inv2R * digi(setup_->chosenRofPhi(), baseUr_), baseUphiT_);
        double zT = digi(z0 + cot * digi(setup_->chosenRofZ(), baseUr_), baseUzT_);
        // convert stubs
        std::vector<Stub*> stubs;
        stubs.reserve(channelAssignment_->numSeedingLayers() + numP);
        for (int layer = 0; layer < numP; layer++) {
          const tt::FrameStub& frameStub = streamsStub[offsetStub + layer][frame];
          const TTStubRef& ttStubRef = frameStub.first;
          if (ttStubRef.isNull())
            continue;
          // parse residuals from tt::Frame and take layerId from tt::TTStubRef
          const bool barrel = setup_->barrel(ttStubRef);
          const int layerIdTracklet = setup_->trackletLayerId(ttStubRef);
          const double basePhi = barrel ? settings_->kphi1() : settings_->kphi(layerIdTracklet);
          const double baseRZ = barrel ? settings_->kz(layerIdTracklet) : settings_->kz();
          const int widthRZ = barrel ? settings_->zresidbits() : settings_->rresidbits();
          TTBV hw(frameStub.second);
          const TTBV hwRZ(hw, widthRZ, 0, true);
          hw >>= widthRZ;
          const TTBV hwPhi(hw, settings_->phiresidbits(), 0, true);
          hw >>= settings_->phiresidbits();
          const int indexLayerId = setup_->indexLayerId(ttStubRef);
          const tt::SensorModule::Type type = setup_->type(ttStubRef);
          const int widthR = setup_->tbWidthR(type);
          const double baseR = setup_->hybridBaseR(type);
          const TTBV hwR(hw, widthR, 0, barrel);
          hw >>= widthR;
          const TTBV hwStubId(hw, channelAssignment_->tmWidthStubId(), 0, false);
          const int stubId = hwStubId.val();
          double r = hwR.val(baseR) + (barrel ? setup_->hybridLayerR(indexLayerId) : 0.);
          if (type == tt::SensorModule::Disk2S)
            r = setup_->disk2SR(indexLayerId, r);
          r = digi(r - setup_->chosenRofPhi(), baseUr_);
          double phi = hwPhi.val(basePhi);
          if (basePhi > baseUphi_)
            phi += baseUphi_ / 2.;
          double z = digi(hwRZ.val(baseRZ) * (barrel ? 1. : -cot), baseUz_);
          // determine module type
          bool psTilt = setup_->psModule(ttStubRef);
          if (barrel) {
            const double posZ = (r + digi(setup_->chosenRofPhi(), baseUr_)) * cot + z0 + z;
            const int indexLayerId = setup_->indexLayerId(ttStubRef);
            const double limit = setup_->tiltedLayerLimitZ(indexLayerId);
            psTilt = std::abs(posZ) < limit;
          }
          stubs_.emplace_back(ttStubRef, layerIdTracklet, stubId, r, phi, z, psTilt);
          stubs.push_back(&stubs_.back());
        }
        if (setup_->kfUseTTStubParameters()) {
          std::vector<TTStubRef> seedTTStubRefs;
          seedTTStubRefs.reserve(channelAssignment_->numSeedingLayers());
          std::map<int, TTStubRef> mapStubs;
          for (TTStubRef& ttStubRef : ttTrackRef->getStubRefs())
            mapStubs.emplace(setup_->layerId(ttStubRef), ttStubRef);
          for (int layer : channelAssignment_->seedingLayers(ttTrackRef->trackSeedType()))
            seedTTStubRefs.push_back(mapStubs[layer]);
          const GlobalPoint gp0 = setup_->stubPos(seedTTStubRefs[0]);
          const GlobalPoint gp1 = setup_->stubPos(seedTTStubRefs[1]);
          const double dH = gp1.perp() - gp0.perp();
          const double H1m0 = (gp1.perp() - setup_->chosenRofPhi()) * tt::deltaPhi(gp0.phi() - offset);
          const double H0m1 = (gp0.perp() - setup_->chosenRofPhi()) * tt::deltaPhi(gp1.phi() - offset);
          const double H3m2 = (gp1.perp() - setup_->chosenRofZ()) * gp0.z();
          const double H2m3 = (gp0.perp() - setup_->chosenRofZ()) * gp1.z();
          const double dinv2R = inv2R - (gp1.phi() - gp0.phi()) / dH;
          const double dcot = cot - (gp1.z() - gp0.z()) / dH;
          const double dphiT = phiT - (H1m0 - H0m1) / dH;
          const double dzT = zT - (H3m2 - H2m3) / dH;
          inv2R -= dinv2R;
          cot -= dcot;
          phiT -= dphiT;
          zT -= dzT;
          z0 = zT - cot * setup_->chosenRofZ();
          // adjust stub residuals by track parameter shifts
          for (Stub* stub : stubs) {
            const double dphi = digi(dphiT + stub->r_ * dinv2R, baseUphi_);
            const double r = stub->r_ + digi(setup_->chosenRofPhi() - setup_->chosenRofZ(), baseUr_);
            const double dz = digi(dzT + r * dcot, baseUz_);
            stub->phi_ = digi(stub->phi_ + dphi, baseUphi_);
            stub->z_ = digi(stub->z_ + dz, baseUz_);
          }
        }
        // create fake seed stubs, since TrackBuilder doesn't output these stubs, required by the KF.
        for (int seedingLayer = 0; seedingLayer < channelAssignment_->numSeedingLayers(); seedingLayer++) {
          const int channelStub = numP + seedingLayer;
          const tt::FrameStub& frameStub = streamsStub[offsetStub + channelStub][frame];
          const TTStubRef& ttStubRef = frameStub.first;
          const int trackletLayerId = setup_->trackletLayerId(ttStubRef);
          const int layerId = channelAssignment_->layerId(channel, channelStub);
          const int stubId = TTBV(frameStub.second).val(channelAssignment_->tmWidthStubId());
          const bool barrel = setup_->barrel(ttStubRef);
          double r;
          if (barrel) {
            const int index = layerId - setup_->offsetLayerId();
            const double layer = digi(setup_->hybridLayerR(index), baseUr_);
            const double z = digi(z0 + layer * cot, baseUz_);
            if (std::abs(z) < digi(setup_->tbBarrelHalfLength(), baseUz_) || index > 0)
              r = digi(setup_->hybridLayerR(index) - setup_->chosenRofPhi(), baseUr_);
            else {
              r = digi(setup_->innerRadius() - setup_->chosenRofPhi(), baseUr_);
            }
          } else {
            const int index = layerId - setup_->offsetLayerId() - setup_->offsetLayerDisks();
            const double side = cot < 0. ? -1. : 1.;
            const double disk = digi(setup_->hybridDiskZ(index), baseUzT_);
            const double invCot = digi(1. / digi(std::abs(cot), baseScot_), baseInvCot_);
            const double offset = digi(setup_->chosenRofPhi(), baseUr_);
            r = digi((disk - side * z0) * invCot - offset, baseUr_);
          }
          double phi = 0.;
          double z = 0.;
          // determine module type
          bool psTilt;
          if (barrel) {
            const int indexLayerId = setup_->indexLayerId(ttStubRef);
            const double limit = digi(setup_->tiltedLayerLimitZ(indexLayerId), baseUz_);
            const double posR = digi(setup_->hybridLayerR(layerId - setup_->offsetLayerId()), baseUr_);
            const double posZ = digi(posR * cot + z0, baseUz_);
            psTilt = std::abs(posZ) < limit;
          } else
            psTilt = true;
          stubs_.emplace_back(ttStubRef, trackletLayerId, stubId, r, phi, z, psTilt);
          stubs.push_back(&stubs_.back());
        }
        if (setup_->kfUseTTStubResiduals()) {
          for (Stub* stub : stubs) {
            const GlobalPoint gp = setup_->stubPos(stub->ttStubRef_);
            stub->r_ = gp.perp() - setup_->chosenRofPhi();
            stub->phi_ = tt::deltaPhi(gp.phi() - region_ * setup_->baseRegion());
            stub->phi_ -= phiT + stub->r_ * inv2R;
            stub->z_ = gp.z() - (z0 + gp.perp() * cot);
          }
        }
        // non linear corrections
        if (setup_->kfApplyNonLinearCorrection()) {
          for (Stub* stub : stubs) {
            const double d = inv2R * (stub->r_ + setup_->chosenRofPhi());
            const double dPhi = std::asin(d) - d;
            stub->phi_ -= dPhi;
            stub->z_ -= dPhi / inv2R * cot;
          }
        }
        // check track validity
        bool valid = true;
        // kill truncated rtacks
        if (setup_->enableTruncation() && frame >= setup_->numFramesHigh())
          valid = false;
        // kill tracks outside of fiducial range
        if (!dataFormats_->format(Variable::phiT, Process::tm).inRange(phiT, true))
          valid = false;
        if (!dataFormats_->format(Variable::zT, Process::tm).inRange(zT, true))
          valid = false;
        // stub range checks
        for (Stub* stub : stubs) {
          if (!dataFormats_->format(Variable::phi, Process::tm).inRange(stub->phi_, true))
            stub->valid_ = false;
          if (!dataFormats_->format(Variable::z, Process::tm).inRange(stub->z_, true))
            stub->valid_ = false;
        }
        // layer check
        std::set<int> layers, layersPS;
        for (Stub* stub : stubs) {
          if (!stub->valid_)
            continue;
          const int layerId = setup_->layerId(stub->ttStubRef_);
          layers.insert(layerId);
          if (setup_->psModule(stub->ttStubRef_))
            layersPS.insert(layerId);
        }
        if (static_cast<int>(layers.size()) < setup_->kfMinLayers() ||
            static_cast<int>(layersPS.size()) < setup_->kfMinLayersPS())
          valid = false;
        // create track
        tracks_.emplace_back(ttTrackRef, valid, channel, inv2R, phiT, cot, zT, stubs);
        input.push_back(&tracks_.back());
      }
    }
  }

  // fill output products
  void TrackMultiplexer::produce(tt::StreamsTrack& streamsTrack, tt::StreamsStub& streamsStub) {
    // base transform into high precision TMTT format
    for (Track& track : tracks_) {
      track.inv2R_ = redigi(track.inv2R_, baseUinv2R_, baseHinv2R_, setup_->widthDSPbu());
      track.phiT_ = redigi(track.phiT_, baseUphiT_, baseHphiT_, setup_->widthDSPbu());
      track.cot_ = redigi(track.cot_, baseUcot_, baseHcot_, setup_->widthDSPbu());
      track.zT_ = redigi(track.zT_, baseUzT_, baseHzT_, setup_->widthDSPbu());
      for (Stub* stub : track.stubs_) {
        stub->r_ = redigi(stub->r_, baseUr_, baseHr_, setup_->widthDSPbu());
        stub->phi_ = redigi(stub->phi_, baseUphi_, baseHphi_, setup_->widthDSPbu());
        stub->z_ = redigi(stub->z_, baseUz_, baseHz_, setup_->widthDSPbu());
      }
    }
    // base transform into TMTT format
    for (Track& track : tracks_) {
      // store track parameter shifts
      const double dinv2R = digi(track.inv2R_ - digi(track.inv2R_, baseLinv2R_), baseHinv2R_);
      const double dphiT = digi(track.phiT_ - digi(track.phiT_, baseLphiT_), baseHphiT_);
      const double dcot = track.cot_ - digi(digi(track.zT_, baseLzT_) / setup_->chosenRofZ(), baseHcot_);
      const double dzT = digi(track.zT_ - digi(track.zT_, baseLzT_), baseHzT_);
      // shift track parameter;
      track.inv2R_ -= dinv2R;
      track.phiT_ -= dphiT;
      track.cot_ -= dcot;
      track.zT_ -= dzT;
      // range checks
      if (!dataFormats_->format(Variable::inv2R, Process::tm).inRange(track.inv2R_, true))
        track.valid_ = false;
      if (!dataFormats_->format(Variable::phiT, Process::tm).inRange(track.phiT_, true))
        track.valid_ = false;
      if (!dataFormats_->format(Variable::zT, Process::tm).inRange(track.zT_, true))
        track.valid_ = false;
      // adjust stub residuals by track parameter shifts
      for (Stub* stub : track.stubs_) {
        const double dphi = digi(dphiT + stub->r_ * dinv2R, baseHphi_);
        const double r = stub->r_ + digi(setup_->chosenRofPhi() - setup_->chosenRofZ(), baseHr_);
        const double dz = digi(dzT + r * dcot, baseHz_);
        stub->phi_ = digi(stub->phi_ + dphi, baseLphi_);
        stub->z_ = digi(stub->z_ + dz, baseLz_);
        // range checks
        if (!dataFormats_->format(Variable::phi, Process::tm).inRange(stub->phi_, true))
          stub->valid_ = false;
        if (!dataFormats_->format(Variable::z, Process::tm).inRange(stub->z_, true))
          stub->valid_ = false;
      }
    }
    // emualte clock domain crossing
    static constexpr int ticksPerGap = 3;
    static constexpr int gapPos = 1;
    std::vector<std::deque<Track*>> streams(channelAssignment_->numChannelsTrack());
    for (int channel = 0; channel < channelAssignment_->numChannelsTrack(); channel++) {
      int iTrack(0);
      std::deque<Track*>& stream = streams[channel];
      const std::vector<Track*>& intput = input_[channel];
      for (int tick = 0; iTrack < (int)intput.size(); tick++) {
        Track* track = tick % ticksPerGap != gapPos ? intput[iTrack++] : nullptr;
        stream.push_back(track && track->valid_ ? track : nullptr);
      }
    }
    // remove all gaps between end and last track
    for (std::deque<Track*>& stream : streams)
      for (auto it = stream.end(); it != stream.begin();)
        it = (*--it) ? stream.begin() : stream.erase(it);
    // route into single channel
    std::deque<Track*> accepted;
    std::vector<std::deque<Track*>> stacks(channelAssignment_->numChannelsTrack());
    // clock accurate firmware emulation, each while trip describes one clock tick, one stub in and one stub out per tick
    while (!std::all_of(streams.begin(), streams.end(), [](const std::deque<Track*>& tracks) {
      return tracks.empty();
    }) || !std::all_of(stacks.begin(), stacks.end(), [](const std::deque<Track*>& tracks) { return tracks.empty(); })) {
      // fill input fifos
      for (int channel = 0; channel < channelAssignment_->numChannelsTrack(); channel++) {
        Track* track = pop_front(streams[channel]);
        if (track)
          stacks[channel].push_back(track);
      }
      // merge input fifos to one stream, prioritizing lower input channel over higher channel, affects DR
      bool nothingToRoute(true);
      for (int channel : channelAssignment_->tmMuxOrder()) {
        Track* track = pop_front(stacks[channel]);
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
    if (setup_->enableTruncation() && static_cast<int>(accepted.size()) > setup_->numFramesHigh())
      accepted.resize(setup_->numFramesHigh());
    // remove all gaps between end and last track
    for (auto it = accepted.end(); it != accepted.begin();)
      it = (*--it) ? accepted.begin() : accepted.erase(it);
    // store helper
    auto frameTrack = [this](Track* track) { return track->valid_ ? track->frame(dataFormats_) : tt::FrameTrack(); };
    auto frameStub = [this](Track* track, int layer) {
      const auto it = std::find_if(
          track->stubs_.begin(), track->stubs_.end(), [layer](Stub* stub) { return stub->layer_ == layer; });
      if (!track->valid_ || it == track->stubs_.end() || !(*it)->valid_)
        return tt::FrameStub();

      return (*it)->frame(dataFormats_);
    };
    const int offsetStub = region_ * channelAssignment_->tmNumLayers();
    // fill output tracks and stubs
    streamsTrack[region_].reserve(accepted.size());
    for (int layer = 0; layer < channelAssignment_->tmNumLayers(); layer++)
      streamsStub[offsetStub + layer].reserve(accepted.size());
    for (Track* track : accepted) {
      if (!track) {  // fill gaps
        streamsTrack[region_].emplace_back(tt::FrameTrack());
        for (int layer = 0; layer < channelAssignment_->tmNumLayers(); layer++)
          streamsStub[offsetStub + layer].emplace_back(tt::FrameStub());
        continue;
      }
      streamsTrack[region_].emplace_back(frameTrack(track));
      for (int layer = 0; layer < channelAssignment_->tmNumLayers(); layer++)
        streamsStub[offsetStub + layer].emplace_back(frameStub(track, layer));
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

  // basetransformation of val from baseLow into baseHigh using widthMultiplier bit multiplication
  double TrackMultiplexer::redigi(double val, double baseLow, double baseHigh, int widthMultiplier) const {
    const double base = std::pow(2, 1 - widthMultiplier);
    const double transform = digi(baseLow / baseHigh, base);
    return (std::floor(val * transform / baseLow) + .5) * baseHigh;
  }

}  // namespace trklet
