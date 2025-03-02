#include "L1Trigger/TrackerTFP/interface/CleanTrackBuilder.h"

#include <numeric>
#include <algorithm>
#include <iterator>
#include <deque>
#include <vector>
#include <cmath>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  CleanTrackBuilder::CleanTrackBuilder(const Setup* setup,
                                       const DataFormats* dataFormats,
                                       const LayerEncoding* layerEncoding,
                                       const DataFormat& cot,
                                       vector<StubCTB>& stubs,
                                       vector<TrackCTB>& tracks)
      : setup_(setup),
        dataFormats_(dataFormats),
        layerEncoding_(layerEncoding),
        cot_(cot),
        stubsCTB_(stubs),
        tracksCTB_(tracks),
        r_(dataFormats_->format(Variable::r, Process::ctb)),
        phi_(dataFormats_->format(Variable::phi, Process::ctb)),
        z_(dataFormats_->format(Variable::z, Process::ctb)),
        phiT_(dataFormats_->format(Variable::phiT, Process::ctb)),
        zT_(dataFormats_->format(Variable::zT, Process::ctb)) {
    stubs_.reserve(stubs.capacity());
    tracks_.reserve(tracks.capacity());
    numChannelOut_ = dataFormats_->numChannel(Process::ctb);
    numChannel_ = dataFormats_->numChannel(Process::ht) / numChannelOut_;
    numLayers_ = setup_->numLayers();
    wlayer_ = dataFormats_->width(Variable::layer, Process::ctb);
    numBinsInv2R_ = setup_->ctbNumBinsInv2R();
    numBinsPhiT_ = setup_->ctbNumBinsPhiT();
    numBinsCot_ = setup_->ctbNumBinsCot();
    numBinsZT_ = setup_->ctbNumBinsZT();
    baseInv2R_ = dataFormats_->base(Variable::inv2R, Process::ctb) / numBinsInv2R_;
    basePhiT_ = phiT_.base() / numBinsPhiT_;
    baseCot_ = cot_.base();
    baseZT_ = zT_.base() / numBinsZT_;
  }

  // fill output products
  void CleanTrackBuilder::produce(const vector<vector<StubHT*>>& streamsIn,
                                  vector<deque<TrackCTB*>>& regionTracks,
                                  vector<vector<deque<StubCTB*>>>& regionStubs) {
    // loop over worker
    for (int channelOut = 0; channelOut < numChannelOut_; channelOut++) {
      // clean input tracks
      vector<deque<Track*>> streamsT(numChannel_);
      vector<deque<Stub*>> streamsS(numChannel_);
      for (int cin = 0; cin < numChannel_; cin++) {
        //if (cin != 1)
        //continue;
        const int index = numChannel_ * cin + channelOut;
        cleanStream(streamsIn[index], streamsT[cin], streamsS[cin], index);
      }
      // route
      deque<Track*> tracks;
      vector<deque<Stub*>> stubs(numLayers_);
      route(streamsT, tracks);
      route(streamsS, stubs);
      // sort
      sort(tracks, stubs);
      // convert
      deque<TrackCTB*>& channelTracks = regionTracks[channelOut];
      vector<deque<StubCTB*>>& channelStubs = regionStubs[channelOut];
      convert(tracks, stubs, channelTracks, channelStubs);
    }
  }

  //
  void CleanTrackBuilder::cleanStream(const vector<StubHT*>& input,
                                      deque<Track*>& tracks,
                                      deque<Stub*>& stubs,
                                      int channelId) {
    const DataFormat& dfInv2R = dataFormats_->format(Variable::inv2R, Process::ht);
    const double inv2R = dfInv2R.floating(dfInv2R.toSigned(channelId));
    const int offset = channelId * setup_->ctbMaxTracks();
    int trackId = offset;
    // identify tracks in input container
    int id;
    auto toTrkId = [this](StubHT* stub) {
      const DataFormat& phiT = dataFormats_->format(Variable::phiT, Process::ht);
      const DataFormat& zT = dataFormats_->format(Variable::zT, Process::ht);
      return (phiT.ttBV(stub->phiT()) + zT.ttBV(stub->zT())).val();
    };
    auto different = [&id, toTrkId](StubHT* stub) { return id != toTrkId(stub); };
    int delta = -setup_->htMinLayers() + 1;
    int old = 0;
    for (auto it = input.begin(); it != input.end();) {
      id = toTrkId(*it);
      const auto start = it;
      const auto end = find_if(start, input.end(), different);
      const vector<StubHT*> track(start, end);
      // restore clock accurancy
      delta += (int)track.size() - old;
      old = track.size();
      if (delta > 0) {
        stubs.insert(stubs.end(), delta, nullptr);
        tracks.insert(tracks.end(), delta, nullptr);
        delta = 0;
      }
      // run single track through r-phi and r-z hough transform
      cleanTrack(track, tracks, stubs, inv2R, (*start)->zT(), trackId++);
      if (trackId - offset == setup_->ctbMaxTracks())
        break;
      // set begin of next track
      it = end;
    }
  }

  // run single track through r-phi and r-z hough transform
  void CleanTrackBuilder::cleanTrack(
      const vector<StubHT*>& track, deque<Track*>& tracks, deque<Stub*>& stubs, double inv2R, int zT, int trackId) {
    const TTBV& maybePattern = layerEncoding_->maybePattern(zT);
    auto noTrack = [this, &maybePattern](const TTBV& pattern) {
      // not enough seeding layer
      if (pattern.count(0, setup_->kfMaxSeedingLayer()) < 2)
        return true;
      int nHits(0);
      int nGaps(0);
      bool doubleGap = false;
      for (int layer = 0; layer < numLayers_; layer++) {
        if (pattern.test(layer)) {
          doubleGap = false;
          if (++nHits == setup_->ctbMinLayers())
            return false;
        } else if (!maybePattern.test(layer)) {
          if (++nGaps == setup_->kfMaxGaps() || doubleGap)
            break;
          doubleGap = true;
        }
      }
      return true;
    };
    auto toLayerId = [this](StubHT* stub) { return stub->layer().val(wlayer_); };
    auto toDPhi = [this, inv2R](StubHT* stub) {
      const bool barrel = stub->layer()[5];
      const bool ps = stub->layer()[4];
      const bool tilt = stub->layer()[3];
      const double pitchRow = ps ? setup_->pitchRowPS() : setup_->pitchRow2S();
      const double pitchCol = ps ? setup_->pitchColPS() : setup_->pitchCol2S();
      const double pitchColR = barrel ? (tilt ? setup_->tiltUncertaintyR() : 0.0) : pitchCol;
      const double r = stub->r() + setup_->chosenRofPhi();
      const double dPhi = pitchRow / r + (setup_->scattering() + pitchColR) * abs(inv2R);
      return phi_.digi(dPhi / 2.);
    };
    auto toDZ = [this](StubHT* stub) {
      const double m = setup_->tiltApproxSlope();
      const double c = setup_->tiltApproxIntercept();
      const bool barrel = stub->layer()[5];
      const bool ps = stub->layer()[4];
      const bool tilt = stub->layer()[3];
      const double pitchCol = ps ? setup_->pitchColPS() : setup_->pitchCol2S();
      const double zT = zT_.floating(stub->zT());
      const double cot = abs(zT) / setup_->chosenRofZ();
      const double dZ = (barrel ? (tilt ? m * cot + c : 1.) : cot) * pitchCol;
      return z_.digi(dZ / 2.);
    };
    vector<Stub*> tStubs;
    tStubs.reserve(track.size());
    vector<TTBV> hitPatternPhi(numBinsInv2R_ * numBinsPhiT_, TTBV(0, numLayers_));
    vector<TTBV> hitPatternZ(numBinsCot_ * numBinsZT_, TTBV(0, numLayers_));
    TTBV tracksPhi(0, numBinsInv2R_ * numBinsPhiT_);
    TTBV tracksZ(0, numBinsCot_ * numBinsZT_);
    // identify finer tracks each stub is consistent with
    for (StubHT* stub : track) {
      const int layerId = toLayerId(stub);
      const double dPhi = toDPhi(stub);
      const double dZ = toDZ(stub);
      // r - phi HT
      auto phiT = [stub](double inv2R, double dPhi) { return inv2R * stub->r() + stub->phi() + dPhi; };
      TTBV hitsPhi(0, numBinsInv2R_ * numBinsPhiT_);
      for (int binInv2R = 0; binInv2R < numBinsInv2R_; binInv2R++) {
        const int offset = binInv2R * numBinsPhiT_;
        const double inv2RMin = (binInv2R - numBinsInv2R_ / 2.) * baseInv2R_;
        const double inv2RMax = inv2RMin + baseInv2R_;
        const auto phiTs = {phiT(inv2RMin, -dPhi), phiT(inv2RMax, -dPhi), phiT(inv2RMin, dPhi), phiT(inv2RMax, dPhi)};
        const int binPhiTMin = floor(*min_element(phiTs.begin(), phiTs.end()) / basePhiT_ + 1.e-11) + numBinsPhiT_ / 2;
        const int binPhiTMax = floor(*max_element(phiTs.begin(), phiTs.end()) / basePhiT_ + 1.e-11) + numBinsPhiT_ / 2;
        for (int binPhiT = 0; binPhiT < numBinsPhiT_; binPhiT++)
          if (binPhiT >= binPhiTMin && binPhiT <= binPhiTMax)
            hitsPhi.set(offset + binPhiT);
      }
      // check for tracks on the fly
      for (int phi : hitsPhi.ids()) {
        hitPatternPhi[phi].set(layerId);
        if (!noTrack(hitPatternPhi[phi]))
          tracksPhi.set(phi);
      }
      // r - z HT
      auto zT = [this, stub](double cot, double dZ) {
        const double r = r_.digi(stub->r() + r_.digi(setup_->chosenRofPhi() - setup_->chosenRofZ()));
        return cot * r + stub->z() + dZ;
      };
      TTBV hitsZ(0, numBinsCot_ * numBinsZT_);
      for (int binCot = 0; binCot < numBinsCot_; binCot++) {
        const int offset = binCot * numBinsZT_;
        const double cotMin = (binCot - numBinsCot_ / 2.) * baseCot_;
        const double cotMax = cotMin + baseCot_;
        const auto zTs = {zT(cotMin, -dZ), zT(cotMax, -dZ), zT(cotMin, dZ), zT(cotMax, dZ)};
        const int binZTMin = floor(*min_element(zTs.begin(), zTs.end()) / baseZT_ + 1.e-11) + numBinsZT_ / 2;
        const int binZTMax = floor(*max_element(zTs.begin(), zTs.end()) / baseZT_ + 1.e-11) + numBinsZT_ / 2;
        for (int binZT = 0; binZT < numBinsZT_; binZT++)
          if (binZT >= binZTMin && binZT <= binZTMax)
            hitsZ.set(offset + binZT);
      }
      // check for tracks on the fly
      for (int z : hitsZ.ids()) {
        hitPatternZ[z].set(layerId);
        if (!noTrack(hitPatternZ[z]))
          tracksZ.set(z);
      }
      // store stubs consistent finer tracks
      stubs_.emplace_back(stub, trackId, hitsPhi, hitsZ, layerId, dPhi, dZ);
      tStubs.push_back(&stubs_.back());
    }
    // clean
    tracks.insert(tracks.end(), tStubs.size() - 1, nullptr);
    tracks_.emplace_back(setup_, trackId, tracksPhi, tracksZ, tStubs, inv2R);
    tracks.push_back(&tracks_.back());
    stubs.insert(stubs.end(), tStubs.begin(), tStubs.end());
  }

  //
  void CleanTrackBuilder::Stub::update(const TTBV& phi, const TTBV& z, vector<int>& ids, int max) {
    auto consistent = [](const TTBV& stub, const TTBV& track) {
      for (int id : track.ids())
        if (stub[id])
          return true;
      return false;
    };
    if (consistent(hitsPhi_, phi) && consistent(hitsZ_, z) && ids[layerId_] < max)
      stubId_ = ids[layerId_]++;
    else
      valid_ = false;
  }

  // construct Track from Stubs
  CleanTrackBuilder::Track::Track(const Setup* setup,
                                  int trackId,
                                  const TTBV& hitsPhi,
                                  const TTBV& hitsZ,
                                  const std::vector<Stub*>& stubs,
                                  double inv2R)
      : valid_(true), stubs_(stubs), trackId_(trackId), hitsPhi_(hitsPhi), hitsZ_(hitsZ), inv2R_(inv2R) {
    vector<int> stubIds(setup->numLayers(), 0);
    for (Stub* stub : stubs_)
      stub->update(hitsPhi_, hitsZ_, stubIds, setup->ctbMaxStubs());
    const int nLayer =
        accumulate(stubIds.begin(), stubIds.end(), 0, [](int sum, int i) { return sum + (i > 0 ? 1 : 0); });
    if (nLayer < setup->ctbMinLayers())
      valid_ = false;
    size_ = *max_element(stubIds.begin(), stubIds.end());
  }

  //
  void CleanTrackBuilder::route(vector<deque<Stub*>>& input, vector<deque<Stub*>>& outputs) const {
    for (int channelOut = 0; channelOut < (int)outputs.size(); channelOut++) {
      deque<Stub*>& output = outputs[channelOut];
      vector<deque<Stub*>> inputs(input);
      for (deque<Stub*>& stream : inputs) {
        for (Stub*& stub : stream)
          if (stub && (!stub->valid_ || stub->layerId_ != channelOut))
            stub = nullptr;
        for (auto it = stream.end(); it != stream.begin();)
          it = (*--it) ? stream.begin() : stream.erase(it);
      }
      vector<deque<Stub*>> stacks(input.size());
      // clock accurate firmware emulation, each while trip describes one clock tick, one stub in and one stub out per tick
      while (!all_of(inputs.begin(), inputs.end(), [](const deque<Stub*>& stubs) { return stubs.empty(); }) or
             !all_of(stacks.begin(), stacks.end(), [](const deque<Stub*>& stubs) { return stubs.empty(); })) {
        // fill input fifos
        for (int channel = 0; channel < (int)input.size(); channel++) {
          deque<Stub*>& stack = stacks[channel];
          Stub* stub = pop_front(inputs[channel]);
          if (stub) {
            if (setup_->enableTruncation() && (int)stack.size() == setup_->ctbDepthMemory() - 1)
              pop_front(stack);
            stack.push_back(stub);
          }
        }
        // merge input fifos to one stream, prioritizing lower input channel over higher channel
        bool nothingToRoute(true);
        for (int channel = 0; channel < (int)input.size(); channel++) {
          Stub* stub = pop_front(stacks[channel]);
          if (stub) {
            nothingToRoute = false;
            output.push_back(stub);
            break;
          }
        }
        if (nothingToRoute)
          output.push_back(nullptr);
      }
    }
  }

  //
  void CleanTrackBuilder::route(vector<deque<Track*>>& inputs, deque<Track*>& output) const {
    vector<deque<Track*>> stacks(inputs.size());
    // clock accurate firmware emulation, each while trip describes one clock tick, one stub in and one stub out per tick
    while (!all_of(inputs.begin(), inputs.end(), [](const deque<Track*>& tracks) { return tracks.empty(); }) or
           !all_of(stacks.begin(), stacks.end(), [](const deque<Track*>& tracks) { return tracks.empty(); })) {
      // fill input fifos
      for (int channel = 0; channel < (int)inputs.size(); channel++) {
        deque<Track*>& stack = stacks[channel];
        Track* track = pop_front(inputs[channel]);
        if (track && track->valid_) {
          if (setup_->enableTruncation() && (int)stack.size() == setup_->ctbDepthMemory() - 1)
            pop_front(stack);
          stack.push_back(track);
        }
      }
      // merge input fifos to one stream, prioritizing lower input channel over higher channel
      bool nothingToRoute(true);
      for (int channel = 0; channel < (int)inputs.size(); channel++) {
        Track* track = pop_front(stacks[channel]);
        if (track) {
          nothingToRoute = false;
          output.push_back(track);
          break;
        }
      }
      if (nothingToRoute)
        output.push_back(nullptr);
    }
  }

  // sort
  void CleanTrackBuilder::sort(deque<Track*>& tracks, vector<deque<Stub*>>& stubs) const {
    // aplly truncation
    if (setup_->enableTruncation()) {
      if ((int)tracks.size() > setup_->numFramesHigh())
        tracks.resize(setup_->numFramesHigh());
      for (deque<Stub*>& stream : stubs)
        if ((int)stream.size() > setup_->numFramesHigh())
          stream.resize(setup_->numFramesHigh());
    }
    // cycle event, remove all gaps
    tracks.erase(remove(tracks.begin(), tracks.end(), nullptr), tracks.end());
    for (deque<Stub*>& stream : stubs)
      stream.erase(remove(stream.begin(), stream.end(), nullptr), stream.end());
    // prepare sort according to track id arrival order
    vector<int> trackIds;
    trackIds.reserve(tracks.size());
    transform(tracks.begin(), tracks.end(), back_inserter(trackIds), [](Track* track) { return track->trackId_; });
    auto cleaned = [&trackIds](Stub* stub) {
      return find(trackIds.begin(), trackIds.end(), stub->trackId_) == trackIds.end();
    };
    auto order = [&trackIds](auto lhs, auto rhs) {
      const auto l = find(trackIds.begin(), trackIds.end(), lhs->trackId_);
      const auto r = find(trackIds.begin(), trackIds.end(), rhs->trackId_);
      return distance(r, l) < 0;
    };
    for (deque<Stub*>& stream : stubs) {
      // remove stubs from removed tracks
      stream.erase(remove_if(stream.begin(), stream.end(), cleaned), stream.end());
      // sort according to stub id on layer
      stable_sort(stream.begin(), stream.end(), [](Stub* lhs, Stub* rhs) { return lhs->stubId_ < rhs->stubId_; });
      // sort according to track id arrival order
      stable_sort(stream.begin(), stream.end(), order);
    }
    // add all gaps
    const int size =
        accumulate(tracks.begin(), tracks.end(), 0, [](int sum, Track* track) { return sum + track->size_; });
    for (int frame = 0; frame < size;) {
      const int trackId = tracks[frame]->trackId_;
      const int length = tracks[frame]->size_;
      tracks.insert(next(tracks.begin(), frame + 1), length - 1, nullptr);
      for (int layer = 0; layer < numLayers_; layer++) {
        deque<Stub*>& stream = stubs[layer];
        if (frame >= (int)stream.size()) {
          stream.insert(stream.end(), length, nullptr);
          continue;
        }
        const auto begin = next(stream.begin(), frame);
        const auto end = find_if(begin, stream.end(), [trackId](Stub* stub) { return stub->trackId_ != trackId; });
        stream.insert(end, length - distance(begin, end), nullptr);
      }
      frame += length;
    }
  }

  //
  void CleanTrackBuilder::convert(const deque<Track*>& iTracks,
                                  const vector<deque<Stub*>>& iStubs,
                                  deque<TrackCTB*>& oTracks,
                                  vector<deque<StubCTB*>>& oStubs) {
    for (int iFrame = 0; iFrame < (int)iTracks.size();) {
      Track* track = iTracks[iFrame];
      if (!track) {
        oTracks.push_back(nullptr);
        for (deque<StubCTB*>& stubs : oStubs)
          stubs.push_back(nullptr);
        iFrame++;
        continue;
      }
      StubHT* s = nullptr;
      for (int layer = 0; layer < numLayers_; layer++) {
        for (int n = 0; n < track->size_; n++) {
          Stub* stub = iStubs[layer][iFrame + n];
          if (!stub) {
            oStubs[layer].push_back(nullptr);
            continue;
          }
          s = stub->stubHT_;
          const double r = s->r();
          const double phi = s->phi();
          const double z = s->z();
          const double dPhi = stub->dPhi_;
          const double dZ = stub->dZ_;
          stubsCTB_.emplace_back(*s, r, phi, z, dPhi, dZ);
          oStubs[layer].push_back(&stubsCTB_.back());
        }
      }
      const double inv2R = track->inv2R_;
      const double phiT = dataFormats_->format(Variable::phiT, Process::ctb).floating(s->phiT());
      const double zT = dataFormats_->format(Variable::zT, Process::ctb).floating(s->zT());
      tracksCTB_.emplace_back(TTTrackRef(), dataFormats_, inv2R, phiT, zT);
      oTracks.push_back(&tracksCTB_.back());
      oTracks.insert(oTracks.end(), track->size_ - 1, nullptr);
      iFrame += track->size_;
    }
  }

  // remove and return first element of deque, returns nullptr if empty
  template <class T>
  T* CleanTrackBuilder::pop_front(deque<T*>& ts) const {
    T* t = nullptr;
    if (!ts.empty()) {
      t = ts.front();
      ts.pop_front();
    }
    return t;
  }

  void CleanTrackBuilder::put(TrackCTB* track,
                              const vector<vector<StubCTB*>>& stubs,
                              int region,
                              TTTracks& ttTracks) const {
    const double dPhi = dataFormats_->format(Variable::phiT, Process::ctb).range();
    const double invR = -track->inv2R() * 2.;
    const double phi0 = deltaPhi(track->phiT() - track->inv2R() * setup_->chosenRofPhi() + region * dPhi);
    const double zT = track->zT();
    const double cot = zT / setup_->chosenRofZ();
    TTBV hits(0, numLayers_);
    double chi2phi(0.);
    double chi2z(0.);
    const int nStubs = accumulate(
        stubs.begin(), stubs.end(), 0, [](int sum, const vector<StubCTB*>& layer) { return sum += layer.size(); });
    vector<TTStubRef> ttStubRefs;
    ttStubRefs.reserve(nStubs);
    for (int layer = 0; layer < numLayers_; layer++) {
      for (StubCTB* stub : stubs[layer]) {
        hits.set(layer);
        chi2phi += pow(stub->phi(), 2) / pow(stub->dPhi(), 2);
        chi2z += pow(stub->z(), 2) / pow(stub->dZ(), 2);
        ttStubRefs.push_back(stub->frame().first);
      }
    }
    static constexpr int nPar = 4;
    static constexpr double d0 = 0.;
    static constexpr double z0 = 0;
    static constexpr double trkMVA1 = 0.;
    static constexpr double trkMVA2 = 0.;
    static constexpr double trkMVA3 = 0.;
    const int hitPattern = hits.val();
    const double bField = setup_->bField();
    TTTrack<Ref_Phase2TrackerDigi_> ttTrack(
        invR, phi0, cot, z0, d0, chi2phi, chi2z, trkMVA1, trkMVA2, trkMVA3, hitPattern, nPar, bField);
    ttTrack.setStubRefs(ttStubRefs);
    ttTrack.setPhiSector(region);
    ttTracks.emplace_back(ttTrack);
  }

}  // namespace trackerTFP
