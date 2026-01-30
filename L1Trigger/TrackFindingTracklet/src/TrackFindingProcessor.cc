#include "L1Trigger/TrackFindingTracklet/interface/TrackFindingProcessor.h"
#include "L1Trigger/TrackTrigger/interface/StubPtConsistency.h"

#include <numeric>
#include <algorithm>
#include <iterator>
#include <deque>
#include <vector>

namespace trklet {

  TrackFindingProcessor::TrackFindingProcessor(const tt::Setup* setup, const DataFormats* dataFormats)
      : setup_(setup), dataFormats_(dataFormats) {
    bfield_ = setup_->bField();
  }

  //
  TrackFindingProcessor::Track::Track(const tt::FrameTrack& frameTrackKF,
                                      const tt::FrameTrack& frameTrackTQ,
                                      const std::vector<TTStubRef>& ttStubRefs,
                                      const DataFormats* df)
      : ttTrackRef_(frameTrackKF.first), ttStubRefs_(ttStubRefs), valid_(true) {
    partials_.reserve(partial_in);
    const double rangeInvR = -2. * TTTrack_TrackWord::minRinv;
    const double rangePhi0 = -2. * TTTrack_TrackWord::minPhi0;
    const double rangeCot = -2. * TTTrack_TrackWord::minTanl;
    const double rangeZ0 = -2. * TTTrack_TrackWord::minZ0;
    const double rangeD0 = -2. * TTTrack_TrackWord::minD0;
    const double baseInvR = rangeInvR / std::pow(2., TTTrack_TrackWord::TrackBitWidths::kRinvSize);
    const double basePhi0 = rangePhi0 / std::pow(2., TTTrack_TrackWord::TrackBitWidths::kPhiSize);
    const double baseCot = rangeCot / std::pow(2., TTTrack_TrackWord::TrackBitWidths::kTanlSize);
    const double baseZ0 = rangeZ0 / std::pow(2., TTTrack_TrackWord::TrackBitWidths::kZ0Size);
    const double baseD0 = rangeD0 / std::pow(2., TTTrack_TrackWord::TrackBitWidths::kD0Size);
    // convert bits into nice formats
    const tt::Setup* setup = df->setup();
    const TrackKF trackKF(frameTrackKF, df);
    invR_ = -2. * trackKF.inv2R();
    cot_ = trackKF.cot();
    d0_ = ttTrackRef_->d0();
    const TrackTQ trackTQ(frameTrackTQ, df);
    mva_ = trackTQ.mva();
    channel_ = cot_ < 0. ? 1 : 0;
    z0_ = df->format(Variable::zT, Process::kf)
              .digi(trackKF.zT() - cot_ * df->format(Variable::r, Process::kf).digi(setup->chosenRofZ()));
    phi0_ =
        df->format(Variable::phiT, Process::kf)
            .digi(trackKF.phiT() - trackKF.inv2R() * df->format(Variable::r, Process::kf).digi(setup->chosenRofPhi()));
    // base transforms
    invR_ = redigi(invR_, 2. * df->format(Variable::inv2R, Process::kf).base(), baseInvR, setup->widthDSPbu());
    phi0_ = redigi(phi0_, df->format(Variable::phiT, Process::kf).base(), basePhi0, setup->widthDSPbu());
    cot_ = redigi(cot_, df->format(Variable::cot, Process::kf).base(), baseCot, setup->widthDSPbu());
    z0_ = redigi(z0_, df->format(Variable::zT, Process::kf).base(), baseZ0, setup->widthDSPbu());
    chi20_ = trackTQ.chi20();
    chi21_ = trackTQ.chi21();
    // bin chi2s
    const int dof = (trackTQ.reversedHitPattern().count() - 2);
    chi20bin_ = -1;
    for (double d : TTTrack_TrackWord::chi2RPhiBins)
      if (chi20_ >= d * dof)
        chi20bin_++;
      else
        break;
    chi21bin_ = -1;
    for (double d : TTTrack_TrackWord::chi2RZBins)
      if (chi21_ >= d * dof)
        chi21bin_++;
      else
        break;
    // check ranges
    if (std::abs(invR_) > rangeInvR / 2.)
      valid_ = false;
    if (std::abs(phi0_) > rangePhi0 / 2.)
      valid_ = false;
    if (std::abs(cot_) > rangeCot / 2.)
      valid_ = false;
    if (std::abs(z0_) > rangeZ0 / 2.)
      valid_ = false;
    if (std::abs(d0_) > rangeD0 / 2.)
      valid_ = false;
    if (!valid_)
      return;
    // create bit vectors
    std::string hitPattern = trackTQ.reversedHitPattern().str();
    std::reverse(hitPattern.begin(), hitPattern.end());
    // Drop outermost (8th) track layer, as data format foresees only 7 bits.
    hitPattern.erase(0, 1);
    hitPattern_ = TTBV(hitPattern);
    const TTBV other = TTBV(0, 2 * TTTrack_TrackWord::TrackBitWidths::kMVAQualitySize);
    const TTBV chi2bend = TTBV(0, TTTrack_TrackWord::TrackBitWidths::kBendChi2Size);
    const TTBV d0(0., baseD0, TTTrack_TrackWord::TrackBitWidths::kD0Size, true);
    const TTBV valid = TTBV(1, TTTrack_TrackWord::TrackBitWidths::kValidSize);
    const TTBV mva(mva_, TTTrack_TrackWord::TrackBitWidths::kMVAQualitySize);
    const TTBV chi21(chi21bin_, TTTrack_TrackWord::TrackBitWidths::kChi2RZSize);
    const TTBV z0(z0_, baseZ0, TTTrack_TrackWord::TrackBitWidths::kZ0Size, true);
    const TTBV tanL(cot_, baseCot, TTTrack_TrackWord::TrackBitWidths::kTanlSize, true);
    const TTBV chi20(chi20bin_, TTTrack_TrackWord::TrackBitWidths::kChi2RPhiSize);
    const TTBV phi0(phi0_, basePhi0, TTTrack_TrackWord::TrackBitWidths::kPhiSize, true);
    const TTBV invR(invR_, baseInvR, TTTrack_TrackWord::TrackBitWidths::kRinvSize, true);
    // create partial tt track words
    partials_.emplace_back((valid + invR + phi0 + chi20).str());
    partials_.emplace_back((tanL + z0 + chi21).str());
    partials_.emplace_back((d0 + chi2bend + hitPattern_ + mva + other).str());
  }

  // fill output products
  void TrackFindingProcessor::produce(const tt::StreamsTrack& tracks,
                                      const tt::StreamsStub& stubs,
                                      tt::TTTracks& ttTracks,
                                      tt::StreamsTrack& outputs) {
    // organize input tracks
    std::vector<std::deque<Track*>> streams(outputs.size());
    consume(tracks, stubs, streams);
    // cycle event, remove all gaps
    for (std::deque<Track*>& stream : streams)
      stream.erase(std::remove(stream.begin(), stream.end(), nullptr), stream.end());
    // emualte data format f/w
    produce(streams, outputs);
    // produce TTTracks
    produce(outputs, ttTracks);
  }

  //
  void TrackFindingProcessor::consume(const tt::StreamsTrack& inputs,
                                      const tt::StreamsStub& stubs,
                                      std::vector<std::deque<Track*>>& outputs) {
    // count input objects
    int nTracks(0);
    auto valid = [](int sum, const tt::FrameTrack& frame) { return sum + (frame.first.isNonnull() ? 1 : 0); };
    for (const tt::StreamTrack& tracks : inputs)
      nTracks += std::accumulate(tracks.begin(), tracks.end(), 0, valid);
    tracks_.reserve(nTracks / 2);
    // convert input data
    for (int region = 0; region < setup_->numRegions(); region++) {
      const int offsetTQ = region * setup_->tqNumChannel();
      const int offsetTFP = region * setup_->tfpNumChannel();
      const int offsetStub = region * setup_->numLayers();
      const tt::StreamTrack& streamKF = inputs[offsetTQ + 0];
      const tt::StreamTrack& streamTQ = inputs[offsetTQ + 1];
      for (int channel = 0; channel < setup_->tfpNumChannel(); channel++)
        outputs[offsetTFP + channel] = std::deque<Track*>(streamKF.size(), nullptr);
      for (int frame = 0; frame < static_cast<int>(streamKF.size()); frame++) {
        const tt::FrameTrack& frameTrackKF = streamKF[frame];
        const tt::FrameTrack& frameTrackTQ = streamTQ[frame];
        if (frameTrackKF.first.isNull())
          continue;
        std::vector<TTStubRef> ttStubRefs;
        ttStubRefs.reserve(setup_->numLayers());
        for (int layer = 0; layer < setup_->numLayers(); layer++) {
          const TTStubRef& ttStubRef = stubs[offsetStub + layer][frame].first;
          if (ttStubRef.isNonnull())
            ttStubRefs.push_back(ttStubRef);
        }
        tracks_.emplace_back(frameTrackKF, frameTrackTQ, ttStubRefs, dataFormats_);
        Track& track = tracks_.back();
        outputs[offsetTFP + track.channel_][frame] = track.valid_ ? &track : nullptr;
      }
      // remove all gaps between end and last track
      for (int channel = 0; channel < setup_->tfpNumChannel(); channel++) {
        std::deque<Track*> input = outputs[offsetTFP + channel];
        for (auto it = input.end(); it != input.begin();)
          it = (*--it) ? input.begin() : input.erase(it);
      }
    }
  }

  // emualte data format f/w
  void TrackFindingProcessor::produce(std::vector<std::deque<Track*>>& inputs, tt::StreamsTrack& outputs) const {
    // send 2 tracks overs 3 clock ticks
    for (int channel = 0; channel < static_cast<int>(inputs.size()); channel++) {
      std::deque<Track*>& input = inputs[channel];
      std::deque<PartialFrameTrack> stack;
      std::deque<tt::FrameTrack> output;
      // clock accurate firmware emulation, each while trip describes one clock tick, one stub in and one stub out per tick
      while (!input.empty() || !stack.empty()) {
        output.emplace_back(tt::FrameTrack());
        tt::FrameTrack& frame = output.back();
        Track* track = pop_front(input);
        if (track)
          for (const PartialFrame& pf : track->partials_)
            stack.emplace_back(track->ttTrackRef_, pf);
        TTBV ttBV;
        for (int i = 0; i < partial_out; i++) {
          if (stack.empty()) {
            ttBV += TTBV(0, partial_width);
            continue;
          }
          const PartialFrameTrack& pft = stack.front();
          frame.first = pft.first;
          ttBV += TTBV(pft.second.to_string());
          stack.pop_front();
        }
        frame.second = ttBV.bs();
      }
      // scramble data according to specification
      const int size = std::ceil(output.size() / 3.) * 3;
      output.resize(size);
      for (int i = 0; i < size / 3; i++) {
        const TTBV A1(output[i * 3 + 0].second, TTBV::S_, TTBV::S_ / 2);
        const TTBV A2(output[i * 3 + 0].second, TTBV::S_ / 2, 0);
        const TTBV A3(output[i * 3 + 1].second, TTBV::S_, TTBV::S_ / 2);
        const TTBV B1(output[i * 3 + 1].second, TTBV::S_ / 2, 0);
        const TTBV B2(output[i * 3 + 2].second, TTBV::S_, TTBV::S_ / 2);
        const TTBV B3(output[i * 3 + 2].second, TTBV::S_ / 2, 0);
        output[i * 3 + 0].second = (A2 + A3).bs();
        output[i * 3 + 1].second = (B3 + A1).bs();
        output[i * 3 + 2].second = (B1 + B2).bs();
      }
      // perform truncation
      if (setup_->enableTruncation() && static_cast<int>(output.size()) > setup_->numFramesIOHigh())
        output.resize(setup_->numFramesIOHigh());
      outputs[channel] = tt::StreamTrack(output.begin(), output.end());
    }
  }

  // produce TTTracks
  void TrackFindingProcessor::produce(const tt::StreamsTrack& inputs, tt::TTTracks& outputs) const {
    // collect input TTTrackRefs
    std::vector<TTTrackRef> ttTrackRefs;
    ttTrackRefs.reserve(tracks_.size());
    const TTTrack<Ref_Phase2TrackerDigi_>* last = nullptr;
    for (const tt::StreamTrack& stream : inputs) {
      for (const tt::FrameTrack& frame : stream) {
        const TTTrackRef& ttTrackRef = frame.first;
        if (frame.first.isNull() || last == ttTrackRef.get())
          continue;
        last = ttTrackRef.get();
        ttTrackRefs.push_back(ttTrackRef);
      }
    }
    // convert input TTTrackRefs into output TTTracks
    outputs.reserve(ttTrackRefs.size());
    for (const TTTrackRef& ttTrackRef : ttTrackRefs) {
      auto match = [&ttTrackRef](const Track& track) { return track.ttTrackRef_ == ttTrackRef; };
      const auto it = std::find_if(tracks_.begin(), tracks_.end(), match);
      // TTTrack conversion
      const int region = ttTrackRef->phiSector();
      const double aRinv = it->invR_;
      const double aphi = tt::deltaPhi(it->phi0_ + region * setup_->baseRegion());
      const double aTanLambda = it->cot_;
      const double az0 = it->z0_;
      const double ad0 = it->d0_;
      const double aChi2xyfit = it->chi20_;
      const double aChi2zfit = it->chi21_;
      const double trkMVA1 = (TTTrack_TrackWord::tqMVABins[it->mva_]);
      static constexpr double trkMVA2 = 0.;
      static constexpr double trkMVA3 = 0.;
      const unsigned int aHitpattern = it->hitPattern_.val();
      const unsigned int nPar = ttTrackRef->nFitPars();
      outputs.emplace_back(aRinv,
                           aphi,
                           aTanLambda,
                           az0,
                           ad0,
                           aChi2xyfit,
                           aChi2zfit,
                           trkMVA1,
                           trkMVA2,
                           trkMVA3,
                           aHitpattern,
                           nPar,
                           bfield_);
      TTTrack<Ref_Phase2TrackerDigi_>& ttTrack = outputs.back();
      ttTrack.setPhiSector(region);
      ttTrack.setEtaSector(ttTrackRef->etaSector());
      ttTrack.setTrackSeedType(ttTrackRef->trackSeedType());
      ttTrack.setStubRefs(it->ttStubRefs_);
      ttTrack.setStubPtConsistency(StubPtConsistency::getConsistency(
          ttTrack, setup_->trackerGeometry(), setup_->trackerTopology(), bfield_, nPar));
      ttTrack.setTrackWordBits();
    }
  }

  // produce StreamsTrack
  void TrackFindingProcessor::produce(const std::vector<TTTrackRef>& inputs, tt::StreamsTrack& outputs) const {
    int iTrk(-1);
    const TTTrack<Ref_Phase2TrackerDigi_>* last = nullptr;
    for (tt::StreamTrack& stream : outputs) {
      for (tt::FrameTrack& frame : stream) {
        const TTTrackRef& ttTrackRef = frame.first;
        if (ttTrackRef.isNull())
          continue;
        if (last != ttTrackRef.get())
          iTrk++;
        last = ttTrackRef.get();
        frame.first = inputs[iTrk];
      }
    }
  }

  // remove and return first element of deque, returns nullptr if empty
  template <class T>
  T* TrackFindingProcessor::pop_front(std::deque<T*>& ts) const {
    T* t = nullptr;
    if (!ts.empty()) {
      t = ts.front();
      ts.pop_front();
    }
    return t;
  }

  // basetransformation of val from baseHigh into baseLow using widthMultiplier bit multiplication
  double TrackFindingProcessor::Track::redigi(double val, double baseHigh, double baseLow, int widthMultiplier) const {
    const double base = std::pow(2, -widthMultiplier);
    const double transform = (tt::floor(baseHigh / baseLow / base) + .5) * base;
    return (tt::floor(val * transform / baseHigh) + .5) * baseLow;
  }

}  // namespace trklet
