#include "L1Trigger/TrackFindingTracklet/interface/DRin.h"

#include <vector>
#include <deque>
#include <numeric>
#include <algorithm>

using namespace std;
using namespace edm;
using namespace tt;
using namespace trackerTFP;

namespace trklet {

  DRin::DRin(const ParameterSet& iConfig,
             const Setup* setup,
             const DataFormats* dataFormats,
             const LayerEncoding* layerEncoding,
             const ChannelAssignment* channelAssignment,
             const Settings* settings,
             int region)
      : enableTruncation_(iConfig.getParameter<bool>("EnableTruncation")),
        useTTStubResiduals_(iConfig.getParameter<bool>("UseTTStubResiduals")),
        setup_(setup),
        dataFormats_(dataFormats),
        layerEncoding_(layerEncoding),
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
    // KF input format digitisation granularity (identical to TMTT)
    baseLinv2R_ = dataFormats->base(Variable::inv2R, Process::kfin);
    baseLphiT_ = dataFormats->base(Variable::phiT, Process::kfin);
    baseLcot_ = dataFormats->base(Variable::cot, Process::kfin);
    baseLzT_ = dataFormats->base(Variable::zT, Process::kfin);
    baseLr_ = dataFormats->base(Variable::r, Process::kfin);
    baseLphi_ = dataFormats->base(Variable::phi, Process::kfin);
    baseLz_ = dataFormats->base(Variable::z, Process::kfin);
    // Finer granularity (by powers of 2) than the TMTT one. Used to transform from Tracklet to TMTT base.
    baseHinv2R_ = baseLinv2R_ * pow(2, floor(log2(baseUinv2R_ / baseLinv2R_)));
    baseHphiT_ = baseLphiT_ * pow(2, floor(log2(baseUphiT_ / baseLphiT_)));
    baseHcot_ = baseLcot_ * pow(2, floor(log2(baseUcot_ / baseLcot_)));
    baseHzT_ = baseLzT_ * pow(2, floor(log2(baseUzT_ / baseLzT_)));
    baseHr_ = baseLr_ * pow(2, floor(log2(baseUr_ / baseLr_)));
    baseHphi_ = baseLphi_ * pow(2, floor(log2(baseUphi_ / baseLphi_)));
    baseHz_ = baseLz_ * pow(2, floor(log2(baseUz_ / baseLz_)));
    // calculate digitisation granularity used for inverted cot(theta)
    const int baseShiftInvCot = ceil(log2(setup_->outerRadius() / setup_->hybridRangeR())) - setup_->widthDSPbu();
    baseInvCot_ = pow(2, baseShiftInvCot);
  }

  // read in and organize input tracks and stubs
  void DRin::consume(const StreamsTrack& streamsTrack, const StreamsStub& streamsStub) {
    static const double maxCot = sinh(setup_->maxEta()) + setup_->beamWindowZ() / setup_->chosenRofZ();
    static const int unusedMSBcot = floor(log2(baseUcot_ * pow(2, settings_->nbitst()) / (2. * maxCot)));
    static const double baseCot =
        baseUcot_ * pow(2, settings_->nbitst() - unusedMSBcot - 1 - setup_->widthAddrBRAM18());
    const int offsetTrack = region_ * channelAssignment_->numChannelsTrack();
    // count tracks and stubs to reserve container
    int nTracks(0);
    int nStubs(0);
    for (int channel = 0; channel < channelAssignment_->numChannelsTrack(); channel++) {
      const int channelTrack = offsetTrack + channel;
      const int offsetStub = channelAssignment_->offsetStub(channelTrack);
      const StreamTrack& streamTrack = streamsTrack[channelTrack];
      input_[channel].reserve(streamTrack.size());
      for (int frame = 0; frame < (int)streamTrack.size(); frame++) {
        if (streamTrack[frame].first.isNull())
          continue;
        nTracks++;
        for (int layer = 0; layer < channelAssignment_->numProjectionLayers(channel); layer++)
          if (streamsStub[offsetStub + layer][frame].first.isNonnull())
            nStubs++;
      }
    }
    stubs_.reserve(nStubs + nTracks * channelAssignment_->numSeedingLayers());
    tracks_.reserve(nTracks);
    // store tracks and stubs
    for (int channel = 0; channel < channelAssignment_->numChannelsTrack(); channel++) {
      const int channelTrack = offsetTrack + channel;
      const int offsetStub = channelAssignment_->offsetStub(channelTrack);
      const StreamTrack& streamTrack = streamsTrack[channelTrack];
      vector<Track*>& input = input_[channel];
      for (int frame = 0; frame < (int)streamTrack.size(); frame++) {
        const TTTrackRef& ttTrackRef = streamTrack[frame].first;
        if (ttTrackRef.isNull()) {
          input.push_back(nullptr);
          continue;
        }
        //convert track parameter
        const double r2Inv = digi(-ttTrackRef->rInv() / 2., baseUinv2R_);
        const double phi0U =
            digi(tt::deltaPhi(ttTrackRef->phi() - region_ * setup_->baseRegion() + setup_->hybridRangePhi() / 2.),
                 baseUphiT_);
        const double phi0S = digi(phi0U - setup_->hybridRangePhi() / 2., baseUphiT_);
        const double cot = digi(ttTrackRef->tanL(), baseUcot_);
        const double z0 = digi(ttTrackRef->z0(), baseUzT_);
        const double phiT = digi(phi0S + r2Inv * digi(dataFormats_->chosenRofPhi(), baseUr_), baseUphiT_);
        const double zT = digi(z0 + cot * digi(setup_->chosenRofZ(), baseUr_), baseUzT_);
        // kill tracks outside of fiducial range
        if (abs(phiT) > setup_->baseRegion() / 2. || abs(zT) > setup_->hybridMaxCot() * setup_->chosenRofZ() ||
            abs(z0) > setup_->beamWindowZ()) {
          input.push_back(nullptr);
          continue;
        }
        // convert stubs
        vector<Stub*> stubs;
        stubs.reserve(channelAssignment_->numProjectionLayers(channel) + channelAssignment_->numSeedingLayers());
        for (int layer = 0; layer < channelAssignment_->numProjectionLayers(channel); layer++) {
          const FrameStub& frameStub = streamsStub[offsetStub + layer][frame];
          const TTStubRef& ttStubRef = frameStub.first;
          if (ttStubRef.isNull())
            continue;
          const int layerId = channelAssignment_->layerId(channel, layer);
          // parse residuals from tt::Frame and take r and layerId from tt::TTStubRef
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
          const SensorModule::Type type = setup_->type(ttStubRef);
          const int widthR = setup_->tbWidthR(type);
          const double baseR = setup_->hybridBaseR(type);
          const TTBV hwR(hw, widthR, 0, barrel);
          hw >>= widthR;
          double r = hwR.val(baseR) + (barrel ? setup_->hybridLayerR(indexLayerId) : 0.);
          if (type == SensorModule::Disk2S)
            r = setup_->disk2SR(indexLayerId, r);
          r = digi(r - dataFormats_->chosenRofPhi(), baseUr_);
          double phi = hwPhi.val(basePhi);
          if (basePhi > baseUphi_)
            phi += baseUphi_ / 2.;
          const double z = digi(hwRZ.val(baseRZ) * (barrel ? 1. : -cot), baseUz_);
          const TTBV hwStubId(hw, channelAssignment_->widthStubId(), 0, false);
          const int stubId = hwStubId.val();
          // determine module type
          bool psTilt;
          if (barrel) {
            const double posZ = (r + digi(dataFormats_->chosenRofPhi(), baseUr_)) * cot + z0 + z;
            const int indexLayerId = setup_->indexLayerId(ttStubRef);
            const double limit = setup_->tiltedLayerLimitZ(indexLayerId);
            psTilt = abs(posZ) < limit;
          } else
            psTilt = setup_->psModule(ttStubRef);
          if (useTTStubResiduals_) {
            const GlobalPoint gp = setup_->stubPos(ttStubRef);
            const double ttR = r;
            const double ttZ = gp.z() - (z0 + (ttR + dataFormats_->chosenRofPhi()) * cot);
            stubs_.emplace_back(ttStubRef, layerId, layerIdTracklet, false, stubId, ttR, phi, ttZ, psTilt);
          } else
            stubs_.emplace_back(ttStubRef, layerId, layerIdTracklet, false, stubId, r, phi, z, psTilt);
          stubs.push_back(&stubs_.back());
        }
        // create fake seed stubs, since TrackBuilder doesn't output these stubs, required by the KF.
        for (int seedingLayer = 0; seedingLayer < channelAssignment_->numSeedingLayers(); seedingLayer++) {
          const int channelStub = channelAssignment_->numProjectionLayers(channel) + seedingLayer;
          const FrameStub& frameStub = streamsStub[offsetStub + channelStub][frame];
          const TTStubRef& ttStubRef = frameStub.first;
          if (ttStubRef.isNull())
            continue;
          const int layerId = channelAssignment_->layerId(channel, channelStub);
          const int layerIdTracklet = setup_->trackletLayerId(ttStubRef);
          const int stubId = TTBV(frameStub.second).val(channelAssignment_->widthSeedStubId());
          const bool barrel = setup_->barrel(ttStubRef);
          double r;
          if (barrel)
            r = digi(setup_->hybridLayerR(layerId - setup_->offsetLayerId()) - dataFormats_->chosenRofPhi(), baseUr_);
          else {
            r = (z0 +
                 digi(setup_->hybridDiskZ(layerId - setup_->offsetLayerId() - setup_->offsetLayerDisks()), baseUzT_)) *
                digi(1. / digi(abs(cot), baseCot), baseInvCot_);
            r = digi(r - digi(dataFormats_->chosenRofPhi(), baseUr_), baseUr_);
          }
          static constexpr double phi = 0.;
          static constexpr double z = 0.;
          // determine module type
          bool psTilt;
          if (barrel) {
            const double posZ =
                digi(digi(setup_->hybridLayerR(layerId - setup_->offsetLayerId()), baseUr_) * cot + z0, baseUz_);
            const int indexLayerId = setup_->indexLayerId(ttStubRef);
            const double limit = digi(setup_->tiltedLayerLimitZ(indexLayerId), baseUz_);
            psTilt = abs(posZ) < limit;
          } else
            psTilt = true;
          const GlobalPoint gp = setup_->stubPos(ttStubRef);
          const double ttR = gp.perp() - dataFormats_->chosenRofPhi();
          const double ttZ = gp.z() - (z0 + (ttR + dataFormats_->chosenRofPhi()) * cot);
          if (useTTStubResiduals_)
            stubs_.emplace_back(ttStubRef, layerId, layerIdTracklet, true, stubId, ttR, phi, ttZ, psTilt);
          else
            stubs_.emplace_back(ttStubRef, layerId, layerIdTracklet, true, stubId, r, phi, z, psTilt);
          stubs.push_back(&stubs_.back());
        }
        const bool valid = frame < setup_->numFrames() ? true : enableTruncation_;
        tracks_.emplace_back(ttTrackRef, valid, r2Inv, phiT, cot, zT, stubs);
        input.push_back(&tracks_.back());
      }
    }
  }

  // fill output products
  void DRin::produce(StreamsStub& accpetedStubs,
                     StreamsTrack& acceptedTracks,
                     StreamsStub& lostStubs,
                     StreamsTrack& lostTracks) {
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
    // find sector
    for (Track& track : tracks_) {
      const int sectorPhi = track.phiT_ < 0. ? 0 : 1;
      track.phiT_ -= (sectorPhi - .5) * setup_->baseSector();
      int sectorEta(-1);
      for (; sectorEta < setup_->numSectorsEta(); sectorEta++)
        if (track.zT_ < digi(setup_->chosenRofZ() * sinh(setup_->boundarieEta(sectorEta + 1)), baseHzT_))
          break;
      if (sectorEta >= setup_->numSectorsEta() || sectorEta <= -1) {
        track.valid_ = false;
        continue;
      }
      track.cot_ = track.cot_ - digi(setup_->sectorCot(sectorEta), baseHcot_);
      track.zT_ = track.zT_ - digi(setup_->chosenRofZ() * setup_->sectorCot(sectorEta), baseHzT_);
      track.sector_ = sectorPhi * setup_->numSectorsEta() + sectorEta;
    }
    // base transform into TMTT format
    for (Track& track : tracks_) {
      if (!track.valid_)
        continue;
      // store track parameter shifts
      const double dinv2R = digi(track.inv2R_ - digi(track.inv2R_, baseLinv2R_), baseHinv2R_);
      const double dphiT = digi(track.phiT_ - digi(track.phiT_, baseLphiT_), baseHphiT_);
      const double dcot = digi(track.cot_ - digi(track.cot_, baseLcot_), baseHcot_);
      const double dzT = digi(track.zT_ - digi(track.zT_, baseLzT_), baseHzT_);
      // shift track parameter;
      track.inv2R_ = digi(track.inv2R_, baseLinv2R_);
      track.phiT_ = digi(track.phiT_, baseLphiT_);
      track.cot_ = digi(track.cot_, baseLcot_);
      track.zT_ = digi(track.zT_, baseLzT_);
      // range checks
      if (!dataFormats_->format(Variable::inv2R, Process::kfin).inRange(track.inv2R_, true))
        track.valid_ = false;
      if (!dataFormats_->format(Variable::phiT, Process::kfin).inRange(track.phiT_, true))
        track.valid_ = false;
      if (!dataFormats_->format(Variable::cot, Process::kfin).inRange(track.cot_, true))
        track.valid_ = false;
      if (!dataFormats_->format(Variable::zT, Process::kfin).inRange(track.zT_, true))
        track.valid_ = false;
      if (!track.valid_)
        continue;
      // adjust stub residuals by track parameter shifts
      for (Stub* stub : track.stubs_) {
        const double dphi = digi(dphiT + stub->r_ * dinv2R, baseHphi_);
        const double r = stub->r_ + digi(dataFormats_->chosenRofPhi() - setup_->chosenRofZ(), baseHr_);
        const double dz = digi(dzT + r * dcot, baseHz_);
        stub->phi_ = digi(stub->phi_ + dphi, baseLphi_);
        stub->z_ = digi(stub->z_ + dz, baseLz_);
        // range checks
        if (!dataFormats_->format(Variable::phi, Process::kfin).inRange(stub->phi_))
          stub->valid_ = false;
        if (!dataFormats_->format(Variable::z, Process::kfin).inRange(stub->z_))
          stub->valid_ = false;
      }
    }
    // encode layer id
    for (Track& track : tracks_) {
      if (!track.valid_)
        continue;
      const int sectorEta = track.sector_ % setup_->numSectorsEta();
      const int zT = dataFormats_->format(Variable::zT, Process::kfin).toUnsigned(track.zT_);
      const int cot = dataFormats_->format(Variable::cot, Process::kfin).toUnsigned(track.cot_);
      for (Stub* stub : track.stubs_) {
        if (!stub->valid_)
          continue;
        // store encoded layerId
        stub->layerKF_ = layerEncoding_->layerIdKF(sectorEta, zT, cot, stub->layer_);
        // kill stubs from layers which can't be crossed by track
        if (stub->layerKF_ == -1)
          stub->valid_ = false;
      }
      TTBV hitPattern(0, setup_->numLayers());
      // kill multiple stubs from same kf layer
      for (Stub* stub : track.stubs_) {
        if (!stub->valid_)
          continue;
        if (hitPattern[stub->layerKF_])
          stub->valid_ = false;
        else
          hitPattern.set(stub->layerKF_);
      }
      // lookup maybe layers
      track.maybe_ = layerEncoding_->maybePattern(sectorEta, zT, cot);
    }
    // kill tracks with not enough layer
    for (Track& track : tracks_) {
      if (!track.valid_)
        continue;
      TTBV hits(0, setup_->numLayers());
      for (const Stub* stub : track.stubs_)
        if (stub->valid_)
          hits.set(stub->layerKF_);
      if (hits.count() < setup_->kfMinLayers())
        track.valid_ = false;
    }
    // store helper
    auto frameTrack = [this](Track* track) {
      const TTBV maybe(track->maybe_);
      const TTBV sectorPhi(
          dataFormats_->format(Variable::sectorPhi, Process::kfin).ttBV(track->sector_ / setup_->numSectorsEta()));
      const TTBV sectorEta(
          dataFormats_->format(Variable::sectorEta, Process::kfin).ttBV(track->sector_ % setup_->numSectorsEta()));
      const TTBV inv2R(dataFormats_->format(Variable::inv2R, Process::kfin).ttBV(track->inv2R_));
      const TTBV phiT(dataFormats_->format(Variable::phiT, Process::kfin).ttBV(track->phiT_));
      const TTBV cot(dataFormats_->format(Variable::cot, Process::kfin).ttBV(track->cot_));
      const TTBV zT(dataFormats_->format(Variable::zT, Process::kfin).ttBV(track->zT_));
      return FrameTrack(track->ttTrackRef_,
                        Frame("1" + maybe.str() + sectorPhi.str() + sectorEta.str() + phiT.str() + inv2R.str() +
                              zT.str() + cot.str()));
    };
    auto frameStub = [this](Track* track, int layer) {
      auto equal = [layer](Stub* stub) { return stub->valid_ && stub->layerKF_ == layer; };
      const auto it = find_if(track->stubs_.begin(), track->stubs_.end(), equal);
      if (it == track->stubs_.end() || !(*it)->valid_)
        return FrameStub();
      Stub* stub = *it;
      const TTBV layerId(stub->layerDet_, channelAssignment_->widthLayerId());
      const TTBV stubId(stub->stubId_, channelAssignment_->widthStubId(), true);
      const TTBV r(dataFormats_->format(Variable::r, Process::kfin).ttBV(stub->r_));
      const TTBV phi(dataFormats_->format(Variable::phi, Process::kfin).ttBV(stub->phi_));
      const TTBV z(dataFormats_->format(Variable::z, Process::kfin).ttBV(stub->z_));
      return FrameStub(stub->ttStubRef_,
                       Frame("1" + to_string(stub->psTilt_) + to_string(stub->seed_) + layerId.str() + stubId.str() +
                             r.str() + phi.str() + z.str()));
    };
    // route tracks into pt bins and store result
    const int offsetTrack = region_ * channelAssignment_->numNodesDR();
    for (int nodeDR = 0; nodeDR < channelAssignment_->numNodesDR(); nodeDR++) {
      deque<Track*> accepted;
      deque<Track*> lost;
      vector<deque<Track*>> stacks(channelAssignment_->numChannelsTrack());
      vector<deque<Track*>> inputs(channelAssignment_->numChannelsTrack());
      for (int channel = 0; channel < channelAssignment_->numChannelsTrack(); channel++) {
        for (Track* track : input_[channel]) {
          const bool match = track && channelAssignment_->nodeDR(track->ttTrackRef_) == nodeDR;
          if (match && !track->valid_)
            lost.push_back(track);
          inputs[channel].push_back(match && track->valid_ ? track : nullptr);
        }
      }
      // remove all gaps between end and last track
      for (deque<Track*>& input : inputs)
        for (auto it = input.end(); it != input.begin();)
          it = (*--it) ? input.begin() : input.erase(it);
      // clock accurate firmware emulation, each while trip describes one clock tick, one stub in and one stub out per tick
      while (!all_of(inputs.begin(), inputs.end(), [](const deque<Track*>& tracks) { return tracks.empty(); }) or
             !all_of(stacks.begin(), stacks.end(), [](const deque<Track*>& tracks) { return tracks.empty(); })) {
        // fill input fifos
        for (int channel = 0; channel < channelAssignment_->numChannelsTrack(); channel++) {
          deque<Track*>& stack = stacks[channel];
          Track* track = pop_front(inputs[channel]);
          if (track) {
            if (enableTruncation_ && (int)stack.size() == channelAssignment_->depthMemory() - 1)
              lost.push_back(pop_front(stack));
            stack.push_back(track);
          }
        }
        // merge input fifos to one stream, prioritizing higher input channel over lower channel
        bool nothingToRoute(true);
        for (int channel = channelAssignment_->numChannelsTrack() - 1; channel >= 0; channel--) {
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
      if (enableTruncation_ && (int)accepted.size() > setup_->numFrames()) {
        const auto limit = next(accepted.begin(), setup_->numFrames());
        copy_if(limit, accepted.end(), back_inserter(lost), [](const Track* track) { return track; });
        accepted.erase(limit, accepted.end());
      }
      // remove all gaps between end and last track
      for (auto it = accepted.end(); it != accepted.begin();)
        it = (*--it) ? accepted.begin() : accepted.erase(it);
      // fill products StreamsStub& accpetedStubs, StreamsTrack& acceptedTracks, StreamsStub& lostStubs, StreamsTrack& lostTracks
      const int channelTrack = offsetTrack + nodeDR;
      const int offsetStub = channelTrack * setup_->numLayers();
      // fill lost tracks and stubs without gaps
      lostTracks[channelTrack].reserve(lost.size());
      for (int layer = 0; layer < setup_->numLayers(); layer++)
        lostStubs[offsetStub + layer].reserve(lost.size());
      for (Track* track : lost) {
        lostTracks[channelTrack].emplace_back(frameTrack(track));
        for (int layer = 0; layer < setup_->numLayers(); layer++)
          lostStubs[offsetStub + layer].emplace_back(frameStub(track, layer));
      }
      // fill accepted tracks and stubs with gaps
      acceptedTracks[channelTrack].reserve(accepted.size());
      for (int layer = 0; layer < setup_->numLayers(); layer++)
        accpetedStubs[offsetStub + layer].reserve(accepted.size());
      for (Track* track : accepted) {
        if (!track) {  // fill gap
          acceptedTracks[channelTrack].emplace_back(FrameTrack());
          for (int layer = 0; layer < setup_->numLayers(); layer++)
            accpetedStubs[offsetStub + layer].emplace_back(FrameStub());
          continue;
        }
        acceptedTracks[channelTrack].emplace_back(frameTrack(track));
        for (int layer = 0; layer < setup_->numLayers(); layer++)
          accpetedStubs[offsetStub + layer].emplace_back(frameStub(track, layer));
      }
    }
  }

  // remove and return first element of deque, returns nullptr if empty
  template <class T>
  T* DRin::pop_front(deque<T*>& ts) const {
    T* t = nullptr;
    if (!ts.empty()) {
      t = ts.front();
      ts.pop_front();
    }
    return t;
  }

  // basetransformation of val from baseLow into baseHigh using widthMultiplier bit multiplication
  double DRin::redigi(double val, double baseLow, double baseHigh, int widthMultiplier) const {
    const double base = pow(2, 1 - widthMultiplier);
    const double transform = digi(baseLow / baseHigh, base);
    return (floor(val * transform / baseLow) + .5) * baseHigh;
  }

}  // namespace trklet