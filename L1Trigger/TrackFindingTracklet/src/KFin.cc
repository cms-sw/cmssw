#include "L1Trigger/TrackFindingTracklet/interface/KFin.h"

#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;
using namespace edm;
using namespace tt;
using namespace trackerTFP;

namespace trklet {

  KFin::KFin(const ParameterSet& iConfig,
             const Setup* setup,
             const DataFormats* dataFormats,
             const LayerEncoding* layerEncoding,
             const ChannelAssignment* channelAssignment,
             int region)
      : enableTruncation_(iConfig.getParameter<bool>("EnableTruncation")),
        setup_(setup),
        dataFormats_(dataFormats),
        layerEncoding_(layerEncoding),
        channelAssignment_(channelAssignment),
        region_(region),
        input_(channelAssignment_->numNodesDR()) {}

  // read in and organize input tracks and stubs
  void KFin::consume(const StreamsTrack& streamsTrack, const StreamsStub& streamsStub) {
    const int offsetTrack = region_ * channelAssignment_->numNodesDR();
    auto nonNullTrack = [](int& sum, const FrameTrack& frame) { return sum += (frame.first.isNonnull() ? 1 : 0); };
    auto nonNullStub = [](int& sum, const FrameStub& frame) { return sum += (frame.first.isNonnull() ? 1 : 0); };
    // count tracks and stubs and reserve corresponding vectors
    int sizeTracks(0);
    int sizeStubs(0);
    for (int channel = 0; channel < channelAssignment_->numNodesDR(); channel++) {
      const int streamTrackId = offsetTrack + channel;
      const int offsetStub = streamTrackId * setup_->numLayers();
      const StreamTrack streamTrack = streamsTrack[streamTrackId];
      input_[channel].reserve(streamTrack.size());
      sizeTracks += accumulate(streamTrack.begin(), streamTrack.end(), 0, nonNullTrack);
      for (int layer = 0; layer < setup_->numLayers(); layer++) {
        const StreamStub streamStub = streamsStub[offsetStub + layer];
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
      const StreamTrack streamTrack = streamsTrack[streamTrackId];
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
          const TTBV zBV(ttBV, dataFormats_->format(Variable::z, Process::kfin).width(), 0, true);
          ttBV >>= dataFormats_->format(Variable::z, Process::kfin).width();
          const TTBV phiBV(ttBV, dataFormats_->format(Variable::phi, Process::kfin).width(), 0, true);
          ttBV >>= dataFormats_->format(Variable::phi, Process::kfin).width();
          const TTBV rBV(ttBV, dataFormats_->format(Variable::r, Process::kfin).width(), 0, true);
          ttBV >>= dataFormats_->format(Variable::r, Process::kfin).width();
          const TTBV layerIdBV(ttBV, channelAssignment_->widthLayerId(), 0);
          ttBV >>= channelAssignment_->widthPSTilt();
          const TTBV tiltBV(ttBV, channelAssignment_->widthPSTilt(), 0);
          const double r = rBV.val(dataFormats_->base(Variable::r, Process::kfin));
          const double phi = phiBV.val(dataFormats_->base(Variable::phi, Process::kfin));
          const double z = zBV.val(dataFormats_->base(Variable::z, Process::kfin));
          stubs_.emplace_back(frameStub.first, r, phi, z, layerIdBV.val(), tiltBV.val(), layer);
          stubs.push_back(&stubs_.back());
        }
        TTBV ttBV = frameTrack.second;
        const TTBV cotBV(ttBV, dataFormats_->format(Variable::cot, Process::kfin).width(), 0, true);
        ttBV >>= dataFormats_->format(Variable::cot, Process::kfin).width();
        const TTBV zTBV(ttBV, dataFormats_->format(Variable::zT, Process::kfin).width(), 0, true);
        ttBV >>= dataFormats_->format(Variable::zT, Process::kfin).width();
        const TTBV phiTBV(ttBV, dataFormats_->format(Variable::phiT, Process::kfin).width(), 0, true);
        ttBV >>= dataFormats_->format(Variable::phiT, Process::kfin).width();
        const TTBV inv2RBV(ttBV, dataFormats_->format(Variable::inv2R, Process::kfin).width(), 0, true);
        ttBV >>= dataFormats_->format(Variable::inv2R, Process::kfin).width();
        const TTBV sectorEtaBV(ttBV, dataFormats_->format(Variable::sectorEta, Process::kfin).width(), 0);
        ttBV >>= dataFormats_->format(Variable::sectorEta, Process::kfin).width();
        const TTBV sectorPhiBV(ttBV, dataFormats_->format(Variable::sectorPhi, Process::kfin).width(), 0);
        const double cot = cotBV.val(dataFormats_->base(Variable::cot, Process::kfin));
        const double zT = zTBV.val(dataFormats_->base(Variable::zT, Process::kfin));
        const double inv2R = inv2RBV.val(dataFormats_->base(Variable::inv2R, Process::kfin));
        const int sectorEta = sectorEtaBV.val();
        const int zTu = dataFormats_->format(Variable::zT, Process::kfin).toUnsigned(zT);
        const int cotu = dataFormats_->format(Variable::cot, Process::kfin).toUnsigned(cot);
        const TTBV maybe = layerEncoding_->maybePattern(sectorEta, zTu, cotu);
        const FrameTrack frameT(frameTrack.first,
                                Frame("1" + maybe.str() + sectorPhiBV.str() + sectorEtaBV.str() + phiTBV.str() +
                                      inv2RBV.str() + zTBV.str() + cotBV.str()));
        tracks_.emplace_back(frameT, stubs, cot, zT, inv2R, sectorEtaBV.val());
        input.push_back(&tracks_.back());
      }
      // remove all gaps between end and last track
      for (auto it = input.end(); it != input.begin();)
        it = (*--it) ? input.begin() : input.erase(it);
    }
  }

  // fill output products
  void KFin::produce(StreamsStub& accpetedStubs,
                     StreamsTrack& acceptedTracks,
                     StreamsStub& lostStubs,
                     StreamsTrack& lostTracks) {
    // calculate stub uncertainties
    static constexpr int usedMSBpitchOverRaddr = 1;
    static const double baseRlut =
        dataFormats_->base(Variable::r, Process::kfin) *
        pow(2, dataFormats_->width(Variable::r, Process::zht) - setup_->widthAddrBRAM18() + usedMSBpitchOverRaddr);
    static const double baseRinvR = dataFormats_->base(Variable::r, Process::kfin) *
                                    pow(2, dataFormats_->width(Variable::r, Process::zht) - setup_->widthAddrBRAM18());
    static const double basePhi =
        dataFormats_->base(Variable::inv2R, Process::kfin) * dataFormats_->base(Variable::r, Process::kfin);
    static const double baseInvR =
        pow(2.,
            ceil(log2(dataFormats_->base(Variable::r, Process::kfin) / setup_->tbInnerRadius())) -
                setup_->widthDSPbu()) /
        dataFormats_->base(Variable::r, Process::kfin);
    static const double maxCot = sinh(setup_->maxEta()) + setup_->beamWindowZ() / setup_->chosenRofZ();
    static constexpr int usedMSBCotLutaddr = 3;
    static const double baseCotLut = pow(2., ceil(log2(maxCot)) - setup_->widthAddrBRAM18() + usedMSBCotLutaddr);
    static const double baseCot = dataFormats_->base(Variable::cot, Process::kfin);
    static const double baseZ = dataFormats_->base(Variable::z, Process::kfin);
    static const double baseR = dataFormats_->base(Variable::r, Process::kfin);
    for (const Track& track : tracks_) {
      const int sectorEta = track.sectorEta_;
      const double inv2R = abs(track.inv2R_);
      for (Stub* stub : track.stubs_) {
        const bool barrel = setup_->barrel(stub->ttStubRef_);
        const bool ps = barrel ? setup_->psModule(stub->ttStubRef_) : stub->psTilt_;
        const bool tilt = barrel ? (ps && !stub->psTilt_) : false;
        const double length = ps ? setup_->lengthPS() : setup_->length2S();
        const double pitch = ps ? setup_->pitchPS() : setup_->pitch2S();
        const double pitchOverR = digi(pitch / (digi(stub->r_, baseRlut) + dataFormats_->chosenRofPhi()), basePhi);
        const double r = digi(stub->r_, baseRinvR) + dataFormats_->chosenRofPhi();
        const double sumdz = track.zT_ + stub->z_;
        const double dZ = digi(sumdz - digi(setup_->chosenRofZ(), baseR) * track.cot_, baseCot * baseR);
        const double sumcot = track.cot_ + digi(setup_->sectorCot(sectorEta), baseCot);
        const double cot = digi(abs(dZ * digi(1. / r, baseInvR) + sumcot), baseCotLut);
        double lengthZ = length;
        double lengthR = 0.;
        if (!barrel) {
          lengthZ = length * cot;
          lengthR = length;
        } else if (tilt) {
          lengthZ = length * abs(setup_->tiltApproxSlope() * cot + setup_->tiltApproxIntercept());
          lengthR = setup_->tiltUncertaintyR();
        }
        const double scat = digi(setup_->scattering(), baseR);
        stub->dZ_ = lengthZ + baseZ;
        stub->dPhi_ = (scat + digi(lengthR, baseR)) * inv2R + pitchOverR;
        stub->dPhi_ = digi(stub->dPhi_, basePhi) + basePhi;
      }
    }
    // store helper
    auto frameTrack = [this](Track* track) { return track->frame_; };
    auto frameStub = [this](Track* track, int layer) {
      auto equal = [layer](Stub* stub) { return stub->channel_ == layer; };
      const auto it = find_if(track->stubs_.begin(), track->stubs_.end(), equal);
      if (it == track->stubs_.end())
        return FrameStub();
      Stub* stub = *it;
      const TTBV r(dataFormats_->format(Variable::r, Process::kfin).ttBV(stub->r_));
      const TTBV phi(dataFormats_->format(Variable::phi, Process::kfin).ttBV(stub->phi_));
      const TTBV z(dataFormats_->format(Variable::z, Process::kfin).ttBV(stub->z_));
      const TTBV dPhi(dataFormats_->format(Variable::dPhi, Process::kfin).ttBV(stub->dPhi_));
      const TTBV dZ(dataFormats_->format(Variable::dZ, Process::kfin).ttBV(stub->dZ_));
      return FrameStub(stub->ttStubRef_, Frame("1" + r.str() + phi.str() + z.str() + dPhi.str() + dZ.str()));
    };
    // merge number of nodes DR to number of Nodes KF and store result
    static const int nMux = channelAssignment_->numNodesDR() / setup_->kfNumWorker();
    const int offsetTrack = region_ * setup_->kfNumWorker();
    for (int nodeKF = 0; nodeKF < setup_->kfNumWorker(); nodeKF++) {
      const int offset = nodeKF * nMux;
      deque<Track*> accepted;
      deque<Track*> lost;
      vector<deque<Track*>> stacks(nMux);
      vector<deque<Track*>> inputs(nMux);
      for (int channel = 0; channel < nMux; channel++) {
        const vector<Track*>& input = input_[offset + channel];
        inputs[channel] = deque<Track*>(input.begin(), input.end());
      }
      // clock accurate firmware emulation, each while trip describes one clock tick, one stub in and one stub out per tick
      while (!all_of(inputs.begin(), inputs.end(), [](const deque<Track*>& tracks) { return tracks.empty(); }) or
             !all_of(stacks.begin(), stacks.end(), [](const deque<Track*>& tracks) { return tracks.empty(); })) {
        // fill input fifos
        for (int channel = 0; channel < nMux; channel++) {
          deque<Track*>& stack = stacks[channel];
          Track* track = pop_front(inputs[channel]);
          if (track)
            stack.push_back(track);
        }
        // merge input fifos to one stream, prioritizing higher input channel over lower channel
        bool nothingToRoute(true);
        for (int channel = nMux - 1; channel >= 0; channel--) {
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
      const int channelTrack = offsetTrack + nodeKF;
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
  T* KFin::pop_front(deque<T*>& ts) const {
    T* t = nullptr;
    if (!ts.empty()) {
      t = ts.front();
      ts.pop_front();
    }
    return t;
  }

}  // namespace trklet
