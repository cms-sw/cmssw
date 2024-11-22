#include "L1Trigger/TrackerTFP/interface/TrackFindingProcessor.h"
#include "L1Trigger/TrackTrigger/interface/StubPtConsistency.h"

#include <numeric>
#include <algorithm>
#include <iterator>
#include <deque>
#include <vector>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  TrackFindingProcessor::TrackFindingProcessor(const ParameterSet& iConfig,
                                               const Setup* setup,
                                               const DataFormats* dataFormats,
                                               const TrackQuality* trackQuality)
      : enableTruncation_(iConfig.getParameter<bool>("EnableTruncation")),
        setup_(setup),
        dataFormats_(dataFormats),
        trackQuality_(trackQuality) {}

  //
  TrackFindingProcessor::Track::Track(const FrameTrack& frameTrack,
                                      const Frame& frameTQ,
                                      const vector<TTStubRef>& ttStubRefs,
                                      const TrackQuality* tq)
      : ttTrackRef_(frameTrack.first), ttStubRefs_(ttStubRefs), valid_(true) {
    partials_.reserve(partial_in);
    // convert bits into nice formats
    const DataFormats* df = tq->dataFormats();
    const Setup* setup = df->setup();
    const TrackDR trackDR(frameTrack, df);
    inv2R_ = trackDR.inv2R();
    phiT_ = trackDR.phiT();
    cot_ = trackDR.cot();
    zT_ = trackDR.zT();
    const double d0 = max(min(ttTrackRef_->d0(), -TTTrack_TrackWord::minD0), TTTrack_TrackWord::minD0);
    TTBV ttBV = TTBV(frameTQ);
    tq->format(VariableTQ::chi2rz).extract(ttBV, chi2rz_);
    tq->format(VariableTQ::chi2rphi).extract(ttBV, chi2rphi_);
    mva_ = TTBV(ttBV, numBinsMVA_).val();
    ttBV >>= numBinsMVA_;
    hitPattern_ = ttBV;
    channel_ = cot_ < 0. ? 0 : 1;
    // convert nice formats into bits
    const double z0 = zT_ - cot_ * setup->chosenRofZ();
    const double phi0 = phiT_ - inv2R_ * setup->chosenRofPhi();
    double invR = -2. * inv2R_;
    if (invR < TTTrack_TrackWord::minRinv)
      invR = TTTrack_TrackWord::minRinv + df->format(Variable::inv2R, Process::dr).base();
    else if (invR > -TTTrack_TrackWord::minRinv)
      invR = -TTTrack_TrackWord::minRinv - df->format(Variable::inv2R, Process::dr).base();
    const double chi2rphi = chi2rphi_ / (hitPattern_.count() - 2);
    const double chi2rz = chi2rz_ / (hitPattern_.count() - 2);
    int chi2rphiBin(-1);
    for (double d : TTTrack_TrackWord::chi2RPhiBins)
      if (chi2rphi >= d)
        chi2rphiBin++;
      else
        break;
    int chi2rzBin(-1);
    for (double d : TTTrack_TrackWord::chi2RZBins)
      if (chi2rz >= d)
        chi2rzBin++;
      else
        break;
    static const double rangeInvR = -2. * TTTrack_TrackWord::minRinv;
    static const double rangePhi0 = -2. * TTTrack_TrackWord::minPhi0;
    static const double rangeCot = -2. * TTTrack_TrackWord::minTanl;
    static const double rangeZ0 = -2. * TTTrack_TrackWord::minZ0;
    static const double rangeD0 = -2. * TTTrack_TrackWord::minD0;
    if (abs(invR) > rangeInvR / 2.)
      valid_ = false;
    if (abs(phi0) > rangePhi0 / 2.)
      valid_ = false;
    if (abs(cot_) > rangeCot / 2.)
      valid_ = false;
    if (abs(z0) > rangeZ0 / 2.)
      valid_ = false;
    if (abs(d0) > rangeD0 / 2.)
      valid_ = false;
    if (!valid_)
      return;
    static const double baseInvR = rangeInvR / pow(2., TTTrack_TrackWord::TrackBitWidths::kRinvSize);
    static const double basePhi0 = rangePhi0 / pow(2., TTTrack_TrackWord::TrackBitWidths::kPhiSize);
    static const double baseCot = rangeCot / pow(2., TTTrack_TrackWord::TrackBitWidths::kTanlSize);
    static const double baseZ0 = rangeZ0 / pow(2., TTTrack_TrackWord::TrackBitWidths::kZ0Size);
    static const double baseD0 = rangeD0 / pow(2., TTTrack_TrackWord::TrackBitWidths::kD0Size);
    static constexpr int nLayers = TTTrack_TrackWord::TrackBitWidths::kHitPatternSize;
    static const TTBV Other_MVAs(0, 2 * TTTrack_TrackWord::TrackBitWidths::kMVAQualitySize);
    const TTBV MVA_quality(mva_, TTTrack_TrackWord::TrackBitWidths::kMVAQualitySize);
    const TTBV hit_pattern(hitPattern_.resize(nLayers).val(), nLayers);
    static const TTBV chi2bend(0, TTTrack_TrackWord::TrackBitWidths::kBendChi2Size);
    static const TTBV D0(d0, baseD0, TTTrack_TrackWord::TrackBitWidths::kD0Size, true);
    const TTBV Chi2rz(chi2rzBin, TTTrack_TrackWord::TrackBitWidths::kChi2RZSize);
    const TTBV Z0(z0, baseZ0, TTTrack_TrackWord::TrackBitWidths::kZ0Size, true);
    const TTBV tanL(cot_, baseCot, TTTrack_TrackWord::TrackBitWidths::kTanlSize, true);
    const TTBV Chi2rphi(chi2rphiBin, TTTrack_TrackWord::TrackBitWidths::kChi2RPhiSize);
    const TTBV Phi0(phi0, basePhi0, TTTrack_TrackWord::TrackBitWidths::kPhiSize, true);
    const TTBV InvR(invR, baseInvR, TTTrack_TrackWord::TrackBitWidths::kRinvSize, true);
    static const TTBV valid(1, TTTrack_TrackWord::TrackBitWidths::kValidSize);
    partials_.emplace_back((valid + InvR + Phi0 + Chi2rphi).str());
    partials_.emplace_back((tanL + Z0 + Chi2rz).str());
    partials_.emplace_back((D0 + chi2bend + hit_pattern + MVA_quality + Other_MVAs).str());
  }

  // fill output products
  void TrackFindingProcessor::produce(const StreamsTrack& inputs,
                                      const StreamsStub& stubs,
                                      TTTracks& ttTracks,
                                      StreamsTrack& outputs) {
    // organize input tracks
    vector<deque<Track*>> streams(outputs.size());
    consume(inputs, stubs, streams);
    // emualte data format f/w
    produce(streams, outputs);
    // produce TTTracks
    produce(outputs, ttTracks);
  }

  //
  void TrackFindingProcessor::consume(const StreamsTrack& inputs,
                                      const StreamsStub& stubs,
                                      vector<deque<Track*>>& outputs) {
    // count input objects
    int nTracks(0);
    auto valid = [](int sum, const FrameTrack& frame) { return sum += (frame.first.isNonnull() ? 1 : 0); };
    for (const StreamTrack& tracks : inputs)
      nTracks += accumulate(tracks.begin(), tracks.end(), 0, valid);
    tracks_.reserve(nTracks);
    // convert input data
    for (int region = 0; region < setup_->numRegions(); region++) {
      const int offsetTQ = region * setup_->tqNumChannel();
      const int offsetTFP = region * setup_->tfpNumChannel();
      const int offsetStub = region * setup_->numLayers();
      const StreamTrack& streamDR = inputs[offsetTQ];
      const StreamTrack& streamTQ = inputs[offsetTQ + 1];
      for (int channel = 0; channel < setup_->tfpNumChannel(); channel++)
        outputs[offsetTFP + channel] = deque<Track*>(streamDR.size(), nullptr);
      for (int frame = 0; frame < (int)streamDR.size(); frame++) {
        const FrameTrack& frameTrack = streamDR[frame];
        const Frame& frameTQ = streamTQ[frame].second;
        if (frameTrack.first.isNull())
          continue;
        vector<TTStubRef> ttStubRefs;
        ttStubRefs.reserve(setup_->numLayers());
        for (int layer = 0; layer < setup_->numLayers(); layer++) {
          const TTStubRef& ttStubRef = stubs[offsetStub + layer][frame].first;
          if (ttStubRef.isNonnull())
            ttStubRefs.push_back(ttStubRef);
        }
        tracks_.emplace_back(frameTrack, frameTQ, ttStubRefs, trackQuality_);
        Track& track = tracks_.back();
        outputs[offsetTFP + track.channel_][frame] = track.valid_ ? &track : nullptr;
      }
      // remove all gaps between end and last track
      for (int channel = 0; channel < setup_->tfpNumChannel(); channel++) {
        deque<Track*> input = outputs[offsetTFP + channel];
        for (auto it = input.end(); it != input.begin();)
          it = (*--it) ? input.begin() : input.erase(it);
      }
    }
  }

  // emualte data format f/w
  void TrackFindingProcessor::produce(vector<deque<Track*>>& inputs, StreamsTrack& outputs) const {
    for (int channel = 0; channel < (int)inputs.size(); channel++) {
      deque<Track*>& input = inputs[channel];
      deque<PartialFrameTrack> stack;
      deque<FrameTrack> output;
      // clock accurate firmware emulation, each while trip describes one clock tick, one stub in and one stub out per tick
      while (!input.empty() || !stack.empty()) {
        output.emplace_back(FrameTrack());
        FrameTrack& frame = output.back();
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
      // perorm truncation
      if (enableTruncation_ && (int)output.size() > setup_->numFramesIOHigh())
        output.resize(setup_->numFramesIOHigh());
      outputs[channel] = StreamTrack(output.begin(), output.end());
    }
  }

  // produce TTTracks
  void TrackFindingProcessor::produce(const StreamsTrack& inputs, TTTracks& outputs) const {
    // collect input TTTrackRefs
    vector<TTTrackRef> ttTrackRefs;
    ttTrackRefs.reserve(tracks_.size());
    const TTTrack<Ref_Phase2TrackerDigi_>* last = nullptr;
    for (const StreamTrack& stream : inputs) {
      for (const FrameTrack& frame : stream) {
        const TTTrackRef& ttTrackRef = frame.first;
        if (frame.first.isNull() || last == ttTrackRef.get())
          continue;
        last = ttTrackRef.get();
        ttTrackRefs.push_back(ttTrackRef);
      }
    }
    // convert input TTTrackRefs into output TTTracks
    static const DataFormat& dfZT = dataFormats_->format(Variable::zT, Process::gp);
    outputs.reserve(ttTrackRefs.size());
    for (const TTTrackRef& ttTrackRef : ttTrackRefs) {
      auto match = [&ttTrackRef](const Track& track) { return track.ttTrackRef_ == ttTrackRef; };
      const auto it = find_if(tracks_.begin(), tracks_.end(), match);
      // TTTrack conversion
      const int region = ttTrackRef->phiSector();
      const double aRinv = -2. * it->inv2R_;
      const double aphi = deltaPhi(it->phiT_ - it->inv2R_ * setup_->chosenRofPhi() + region * setup_->baseRegion());
      const double aTanLambda = it->cot_;
      const double az0 = it->zT_ - it->cot_ * setup_->chosenRofZ();
      const double ad0 = -ttTrackRef->d0();
      const double aChi2xyfit = it->chi2rphi_;
      const double aChi2zfit = it->chi2rz_;
      const double trkMVA1 = (TTTrack_TrackWord::tqMVABins[it->mva_]);
      static constexpr double trkMVA2 = 0.;
      static constexpr double trkMVA3 = 0.;
      const unsigned int aHitpattern = it->hitPattern_.val();
      const unsigned int nPar = ttTrackRef->nFitPars();
      static const double Bfield = setup_->bField();
      outputs.emplace_back(
          aRinv, aphi, aTanLambda, az0, ad0, aChi2xyfit, aChi2zfit, trkMVA1, trkMVA2, trkMVA3, aHitpattern, nPar, Bfield);
      TTTrack<Ref_Phase2TrackerDigi_>& ttTrack = outputs.back();
      ttTrack.setPhiSector(region);
      ttTrack.setEtaSector(dfZT.toUnsigned(dfZT.integer(it->zT_)));
      ttTrack.setTrackSeedType(ttTrackRef->trackSeedType());
      ttTrack.setStubRefs(it->ttStubRefs_);
      ttTrack.setStubPtConsistency(StubPtConsistency::getConsistency(
          ttTrack, setup_->trackerGeometry(), setup_->trackerTopology(), Bfield, nPar));
    }
  }

  // produce StreamsTrack
  void TrackFindingProcessor::produce(const vector<TTTrackRef>& inputs, StreamsTrack& outputs) const {
    int iTrk(-1);
    const TTTrack<Ref_Phase2TrackerDigi_>* last = nullptr;
    for (StreamTrack& stream : outputs) {
      for (FrameTrack& frame : stream) {
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
  T* TrackFindingProcessor::pop_front(deque<T*>& ts) const {
    T* t = nullptr;
    if (!ts.empty()) {
      t = ts.front();
      ts.pop_front();
    }
    return t;
  }

}  // namespace trackerTFP
