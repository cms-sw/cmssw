#include "L1Trigger/TrackFindingTracklet/interface/KalmanFilter.h"

#include <numeric>
#include <algorithm>
#include <iterator>
#include <deque>
#include <vector>
#include <set>
#include <cmath>

namespace trklet {

  KalmanFilter::KalmanFilter(const Setup* setup,
                             const DataFormats* dataFormats,
                             KalmanFilterFormats* kalmanFilterFormats,
                             int region)
      : setup_(setup),
        dataFormats_(dataFormats),
        kalmanFilterFormats_(kalmanFilterFormats),
        region_(region),
        layer_(0) {}

  // read in and organize input tracks and stubs
  void KalmanFilter::consume(const tt::StreamsTrack& streamsTrack, const tt::StreamsStub& streamsStub) {
    auto accT = [](int sum, const tt::FrameTrack& f) { return sum + (f.first.isNull() ? 0 : 1); };
    auto accS = [](int sum, const tt::FrameStub& f) { return sum + (f.first.isNull() ? 0 : 1); };
    const int offset = region_ * setup_->drNumLayers();
    const tt::StreamTrack& streamTrack = streamsTrack[region_];
    const int numTracks = std::accumulate(streamTrack.begin(), streamTrack.end(), 0, accT);
    int numStubs(0);
    for (int layer = 0; layer < setup_->drNumLayers(); layer++) {
      const tt::StreamStub& streamStub = streamsStub[offset + layer];
      numStubs += std::accumulate(streamStub.begin(), streamStub.end(), 0, accS);
    }
    tracks_.reserve(numTracks);
    stubs_.reserve(numStubs);
    int trackId(0);
    for (int frame = 0; frame < static_cast<int>(streamTrack.size()); frame++) {
      const tt::FrameTrack& frameTrack = streamTrack[frame];
      if (frameTrack.first.isNull()) {
        stream_.push_back(nullptr);
        continue;
      }
      tracks_.emplace_back(frameTrack, dataFormats_);
      std::vector<Stub*> seed;
      seed.reserve(setup_->tbNumSeedingLayers());
      std::vector<Stub*> proj;
      proj.reserve(setup_->kfNumProj());
      for (int layer = 0; layer < setup_->drNumLayers(); layer++) {
        const tt::FrameStub& frameStub = streamsStub[offset + layer][frame];
        if (frameStub.first.isNull())
          continue;
        stubs_.emplace_back(kalmanFilterFormats_, frameStub);
        if (layer < setup_->tbNumSeedingLayers())
          seed.push_back(&stubs_.back());
        else
          proj.push_back(&stubs_.back());
      }
      states_.emplace_back(kalmanFilterFormats_, &tracks_.back(), trackId++, seed, proj);
      stream_.push_back(&states_.back());
    }
  }

  // fill output products
  void KalmanFilter::produce(tt::StreamsStub& streamsStub, tt::StreamsTrack& streamsTrack) {
    // calulcate seed parameter
    calcSeeds();
    std::vector<std::deque<State*>> streams(setup_->kfNumProj());
    // Propagate state to each layer in turn, updating it with all viable stub combinations there, using KF maths
    for (layer_ = 0; layer_ < setup_->kfNumProj(); layer_++)
      addLayer(streams[layer_]);
    // apply truncation
    if (setup_->enableTruncation())
      for (std::deque<State*>& stream : streams)
        if (static_cast<int>(stream.size()) > setup_->numFrames())
          stream.resize(setup_->numFrames());
    // cycle event, join streams wihtout gaps
    stream_.clear();
    for (std::deque<State*>& stream : streams)
      for (State* state : stream)
        if (state)
          stream_.push_back(state);
    // apply final cuts
    finalize();
    // best track per candidate selection
    accumulator();
    // Transform States into output products
    conv(streamsStub, streamsTrack);
  }

  // apply final cuts
  void KalmanFilter::finalize() {
    finals_.reserve(stream_.size());
    for (State* state : stream_) {
      int numConsistent(0);
      int numConsistentPS(0);
      std::vector<StubKF> stubsKF;
      stubsKF.reserve(setup_->kfNumLayers());
      auto add = [&numConsistent, &numConsistentPS, &stubsKF, state, this](Stub* stub) {
        const trackerDTC::SensorModule* sm = setup_->sensorModule(stub->stubDR_.frame().first);
        const double dPhi = state->x1() + stub->H() * state->x0() + state->x4() / stub->H();
        const double dZ = state->x3() + stub->H() * state->x2();
        const double phi = digi(VariableKF::m0, stub->m0() - dPhi);
        const double z = digi(VariableKF::m1, stub->m1() - dZ);
        stubsKF.emplace_back(stub->stubDR_, sm->layerIdReduced(), stub->H(), phi, z, stub->d0(), stub->d1());
        if (std::abs(phi) <= stub->d0() && std::abs(z) <= stub->d1()) {
          numConsistent++;
          if (stub->stubDR_.frame().first->moduleTypePS())
            numConsistentPS++;
        }
      };
      for (int layer = 0; layer < setup_->tbNumSeedingLayers(); layer++)
        add(state->seed(layer));
      for (int layer : state->hitPattern().ids())
        add(state->proj(layer));
      // pt cut
      const bool validX0 = dataFormats_->format(Variable::inv2R, Process::kf).isCovered(state->x0());
      // cut on phi sector boundaries
      const bool validX1 = abs(state->x1() + setup_->regChosenRofPhi() * state->x0()) < setup_->regRangePhiT() / 2.;
      // cot cut
      const bool validX2 = dataFormats_->format(Variable::cot, Process::kf).isCovered(state->x2());
      // z0 cut
      const bool validX3 = dataFormats_->format(Variable::z0, Process::kf).isCovered(state->x3());
      if (!validX0 || !validX1 || !validX2 || !validX3)
        continue;
      const TrackKF trackKF(*state->trackDR(), state->x0(), state->x1(), state->x2(), state->x3());
      finals_.emplace_back(state->trackId(), numConsistent, numConsistentPS, state->x4(), trackKF, stubsKF);
    }
  }

  // best state selection
  void KalmanFilter::accumulator() {
    // create container of pointer to make sorts less CPU intense
    std::vector<Track*> finals;
    finals.reserve(finals_.size());
    std::transform(finals_.begin(), finals_.end(), std::back_inserter(finals), [](Track& track) { return &track; });
    // prepare arrival order
    std::vector<int> trackIds;
    trackIds.reserve(tracks_.size());
    for (Track* track : finals) {
      const int trackId = track->trackId_;
      if (std::find_if(trackIds.begin(), trackIds.end(), [trackId](int id) { return id == trackId; }) == trackIds.end())
        trackIds.push_back(trackId);
    }
    // sort in number of consistent stubs
    auto moreConsistentLayers = [](Track* lhs, Track* rhs) { return lhs->numConsistent_ > rhs->numConsistent_; };
    std::stable_sort(finals.begin(), finals.end(), moreConsistentLayers);
    // sort in number of consistent ps stubs
    auto moreConsistentLayersPS = [](Track* lhs, Track* rhs) { return lhs->numConsistentPS_ > rhs->numConsistentPS_; };
    std::stable_sort(finals.begin(), finals.end(), moreConsistentLayersPS);
    // sort in track id as arrived
    auto order = [&trackIds](auto lhs, auto rhs) {
      const auto l = find(trackIds.begin(), trackIds.end(), lhs->trackId_);
      const auto r = find(trackIds.begin(), trackIds.end(), rhs->trackId_);
      return std::distance(r, l) < 0;
    };
    std::stable_sort(finals.begin(), finals.end(), order);
    // keep first state (best due to previous sorts) per track id
    auto sameTrack = [](Track* lhs, Track* rhs) { return lhs->trackId_ == rhs->trackId_; };
    finals.erase(std::unique(finals.begin(), finals.end(), sameTrack), finals.end());
    // apply to actual track container
    int i(0);
    for (Track* track : finals)
      finals_[i++] = *track;
    finals_.resize(i);
  }

  // Transform States into output products
  void KalmanFilter::conv(tt::StreamsStub& streamsStub, tt::StreamsTrack& streamsTrack) {
    const int offset = region_ * setup_->kfNumLayers();
    tt::StreamTrack& streamTrack = streamsTrack[region_];
    streamTrack.reserve(stream_.size());
    for (int layer = 0; layer < setup_->kfNumLayers(); layer++)
      streamsStub[offset + layer].reserve(stream_.size());
    for (const Track& track : finals_) {
      streamTrack.emplace_back(track.trackKF_.frame());
      const std::vector<StubKF>& stubsKF = track.stubsKF_;
      int size = stubsKF.size();
      for (int layer = 0; layer < size; layer++)
        streamsStub[offset + layer].push_back(stubsKF[layer].frame());
      for (int layer = size; layer < setup_->kfNumLayers(); layer++)
        streamsStub[offset + layer].emplace_back();
    }
  }

  // calculates the helix params & their cov. matrix from a pair of stubs
  void KalmanFilter::calcSeeds() {
    auto update = [this](Stub* s) {
      updateRangeActual(VariableKF::m0, s->m0());
      updateRangeActual(VariableKF::m1, s->m1());
      updateRangeActual(VariableKF::v0, s->v0());
      updateRangeActual(VariableKF::v1, s->v1());
      updateRangeActual(VariableKF::H, s->H());
    };
    for (State*& state : stream_) {
      if (!state)
        continue;
      Stub* s1 = state->seed(1);
      Stub* s0 = state->seed(0);
      update(s0);
      update(s1);
      const double dH = digi(VariableKF::dH, s1->H() - s0->H());
      const double invdH = digi(VariableKF::invdH, 1.0 / dH);
      const double invdH2 = digi(VariableKF::invdH2, 1.0 / dH / dH);
      const double H12 = digi(VariableKF::H2, s1->H() * s1->H());
      const double H02 = digi(VariableKF::H2, s0->H() * s0->H());
      const double H1m0 = digi(VariableKF::Hm0, s1->H() * s0->m0());
      const double H0m1 = digi(VariableKF::Hm0, s0->H() * s1->m0());
      const double H3m2 = digi(VariableKF::Hm1, s1->H() * s0->m1());
      const double H2m3 = digi(VariableKF::Hm1, s0->H() * s1->m1());
      const double H1v0 = digi(VariableKF::Hv0, s1->H() * s0->v0());
      const double H0v1 = digi(VariableKF::Hv0, s0->H() * s1->v0());
      const double H3v2 = digi(VariableKF::Hv1, s1->H() * s0->v1());
      const double H2v3 = digi(VariableKF::Hv1, s0->H() * s1->v1());
      const double H12v0 = digi(VariableKF::H2v0, H12 * s0->v0());
      const double H02v1 = digi(VariableKF::H2v0, H02 * s1->v0());
      const double H32v2 = digi(VariableKF::H2v1, H12 * s0->v1());
      const double H22v3 = digi(VariableKF::H2v1, H02 * s1->v1());
      const double x0 = digi(VariableKF::x0, (s1->m0() - s0->m0()) * invdH);
      const double x2 = digi(VariableKF::x2, (s1->m1() - s0->m1()) * invdH);
      const double x1 = digi(VariableKF::x1, (H1m0 - H0m1) * invdH);
      const double x3 = digi(VariableKF::x3, (H3m2 - H2m3) * invdH);
      const double C00 = digi(VariableKF::C00, (s1->v0() + s0->v0()) * invdH2);
      const double C22 = digi(VariableKF::C22, (s1->v1() + s0->v1()) * invdH2);
      const double C01 = -digi(VariableKF::C01, (H1v0 + H0v1) * invdH2);
      const double C23 = -digi(VariableKF::C23, (H3v2 + H2v3) * invdH2);
      const double C11 = digi(VariableKF::C11, (H12v0 + H02v1) * invdH2);
      const double C33 = digi(VariableKF::C33, (H32v2 + H22v3) * invdH2);
      // create updated state
      states_.emplace_back(State(state, {x0, x1, x2, x3, 0., C00, C11, C22, C33, C01, C23, 0., 0., 0.}));
      state = &states_.back();
      updateRangeActual(VariableKF::invdH, invdH);
      updateRangeActual(VariableKF::invdH2, invdH2);
      updateRangeActual(VariableKF::Hv0, H1v0);
      updateRangeActual(VariableKF::Hv0, H0v1);
      updateRangeActual(VariableKF::Hv1, H3v2);
      updateRangeActual(VariableKF::Hv1, H2v3);
      updateRangeActual(VariableKF::H2v0, H12v0);
      updateRangeActual(VariableKF::H2v0, H02v1);
      updateRangeActual(VariableKF::H2v1, H32v2);
      updateRangeActual(VariableKF::H2v1, H22v3);
      updateRangeActual(VariableKF::x0, x0);
      updateRangeActual(VariableKF::x1, x1);
      updateRangeActual(VariableKF::x2, x2);
      updateRangeActual(VariableKF::x3, x3);
      updateRangeActual(VariableKF::C00, C00);
      updateRangeActual(VariableKF::C01, C01);
      updateRangeActual(VariableKF::C11, C11);
      updateRangeActual(VariableKF::C22, C22);
      updateRangeActual(VariableKF::C23, C23);
      updateRangeActual(VariableKF::C33, C33);
    }
  }

  // adds a layer to states
  void KalmanFilter::addLayer(std::deque<State*>& stream) {
    if (stream_.empty())
      return;
    // Latency of KF Associator block firmware
    static constexpr int latency = 5;
    // dynamic state container for clock accurate emulation
    std::deque<State*> streamOutput;
    // Memory stack used to handle combinatorics
    std::deque<State*> stack;
    // static delay container
    std::deque<State*> delay(latency, nullptr);
    // each trip corresponds to a f/w clock tick
    // done if no states to process left, taking as much time as needed
    while (!stream_.empty() || !stack.empty() ||
           !std::all_of(delay.begin(), delay.end(), [](const State* state) { return state == nullptr; })) {
      State* state = pop_front(stream_);
      // Process a combinatoric state if no (non-combinatoric?) state available
      if (!state)
        state = pop_front(stack);
      streamOutput.push_back(state);
      // The remainder of the code in this loop deals with combinatoric states.
      if (state)
        state = state->comb(states_, layer_);
      delay.push_back(state);
      state = pop_front(delay);
      if (state)
        stack.push_back(state);
    }
    stream_ = streamOutput;
    // Update state with next stub using KF maths
    for (State*& state : stream_)
      if (state && state->hitPattern().test(layer_))
        update(state);
    // extract final states
    for (State*& state : stream_) {
      if (!state) {
        stream.push_back(nullptr);
        continue;
      }
      stream.push_back(state->final(states_, layer_));
      if (stream.back() && state->trackPattern().pmEncode() == layer_)
        state = nullptr;
    }
    // remove all gaps between end and last state
    for (auto it = stream.end(); it != stream.begin();)
      it = (*--it) ? stream.begin() : stream.erase(it);
    for (auto it = stream_.end(); it != stream_.begin();)
      it = (*--it) ? stream_.begin() : stream_.erase(it);
  }

  // updates state
  void KalmanFilter::update(State*& state) {
    // All variable names & equations come from Fruhwirth KF paper http://dx.doi.org/10.1016/0168-9002%2887%2990887-4", where F taken as unit matrix. Stub uncertainties projected onto (phi,z), assuming no correlations between r-phi & r-z planes.
    Stub* stub = state->proj(layer_);
    // stub phi residual wrt input helix
    const double m0 = stub->m0();
    // stub z residual wrt input helix
    const double m1 = stub->m1();
    // stub projected phi uncertainty squared);
    const double v0 = stub->v0();
    // stub projected z uncertainty squared
    const double v1 = stub->v1();
    // stub radius
    const double H = stub->H();
    updateRangeActual(VariableKF::m0, m0);
    updateRangeActual(VariableKF::m1, m1);
    updateRangeActual(VariableKF::v0, v0);
    updateRangeActual(VariableKF::v1, v1);
    updateRangeActual(VariableKF::H, H);
    // helix inv2R
    double x0 = state->x0();
    // helix phi0 wrt region center
    double x1 = state->x1();
    // helix cot(Theta)
    double x2 = state->x2();
    // helix z0
    double x3 = state->x3();
    // cov. matrix
    double C00 = state->C00();
    double C01 = state->C01();
    double C11 = state->C11();
    double C22 = state->C22();
    double C23 = state->C23();
    double C33 = state->C33();
    // stub phi residual wrt current state
    const double r0C = digi(VariableKF::x1, m0 - x1);
    const double r0 = digi(VariableKF::r0, r0C - x0 * H);
    // stub z residual wrt current state
    const double r1C = digi(VariableKF::x3, m1 - x3);
    const double r1 = digi(VariableKF::r1, r1C - x2 * H);
    // matrix S = H*C
    const double S00 = digi(VariableKF::S00, C01 + H * C00);
    const double S01 = digi(VariableKF::S01, C11 + H * C01);
    const double S12 = digi(VariableKF::S12, C23 + H * C22);
    const double S13 = digi(VariableKF::S13, C33 + H * C23);
    // Cov. matrix of predicted residuals R = V+HCHt = C+H*St
    const double R00 = digi(VariableKF::R00, v0 + S01 + H * S00);
    const double R11 = digi(VariableKF::R11, v1 + S13 + H * S12);
    // improved dynamic cancelling
    const int msb0 = std::max(0, static_cast<int>(std::ceil(std::log2(R00 / base(VariableKF::R00)))));
    const int msb1 = std::max(0, static_cast<int>(std::ceil(std::log2(R11 / base(VariableKF::R11)))));
    const int shift0 = width(VariableKF::R00) - msb0;
    const int shift1 = width(VariableKF::R11) - msb1;
    const double R00Shifted = R00 * std::pow(2., shift0);
    const double R11Shifted = R11 * std::pow(2., shift1);
    const double R00Rough = digi(VariableKF::R00Rough, R00Shifted);
    const double R11Rough = digi(VariableKF::R11Rough, R11Shifted);
    const double invR00Approx = digi(VariableKF::invR00Approx, 1. / R00Rough);
    const double invR11Approx = digi(VariableKF::invR11Approx, 1. / R11Rough);
    const double invR00Cor = digi(VariableKF::invR00Cor, 2. - invR00Approx * R00Shifted);
    const double invR11Cor = digi(VariableKF::invR11Cor, 2. - invR11Approx * R11Shifted);
    const double invR00 = digi(VariableKF::invR00, invR00Approx * invR00Cor);
    const double invR11 = digi(VariableKF::invR11, invR11Approx * invR11Cor);
    // shift S to "undo" shifting of R
    auto digiShifted = [](double val, double base) { return tt::floor(val / base) * base; };
    const double S00Shifted = digiShifted(S00 * std::pow(2., shift0), base(VariableKF::S00Shifted));
    const double S01Shifted = digiShifted(S01 * std::pow(2., shift0), base(VariableKF::S01Shifted));
    const double S12Shifted = digiShifted(S12 * std::pow(2., shift1), base(VariableKF::S12Shifted));
    const double S13Shifted = digiShifted(S13 * std::pow(2., shift1), base(VariableKF::S13Shifted));
    // Kalman gain matrix K = S*R(inv)
    const double K00 = digi(VariableKF::K00, S00Shifted * invR00);
    const double K10 = digi(VariableKF::K10, S01Shifted * invR00);
    const double K21 = digi(VariableKF::K21, S12Shifted * invR11);
    const double K31 = digi(VariableKF::K31, S13Shifted * invR11);
    // Updated helix params, their cov. matrix
    x0 = digi(VariableKF::x0, x0 + r0 * K00);
    x1 = digi(VariableKF::x1, x1 + r0 * K10);
    x2 = digi(VariableKF::x2, x2 + r1 * K21);
    x3 = digi(VariableKF::x3, x3 + r1 * K31);
    C00 = digi(VariableKF::C00, C00 - S00 * K00);
    C01 = digi(VariableKF::C01, C01 - S01 * K00);
    C11 = digi(VariableKF::C11, C11 - S01 * K10);
    C22 = digi(VariableKF::C22, C22 - S12 * K21);
    C23 = digi(VariableKF::C23, C23 - S13 * K21);
    C33 = digi(VariableKF::C33, C33 - S13 * K31);
    // update variable ranges to tune variable granularity
    updateRangeActual(VariableKF::r0, r0);
    updateRangeActual(VariableKF::r1, r1);
    updateRangeActual(VariableKF::S00, S00);
    updateRangeActual(VariableKF::S01, S01);
    updateRangeActual(VariableKF::S12, S12);
    updateRangeActual(VariableKF::S13, S13);
    updateRangeActual(VariableKF::S00Shifted, S00Shifted);
    updateRangeActual(VariableKF::S01Shifted, S01Shifted);
    updateRangeActual(VariableKF::S12Shifted, S12Shifted);
    updateRangeActual(VariableKF::S13Shifted, S13Shifted);
    updateRangeActual(VariableKF::R00, R00);
    updateRangeActual(VariableKF::R11, R11);
    updateRangeActual(VariableKF::R00Rough, R00Rough);
    updateRangeActual(VariableKF::R11Rough, R11Rough);
    updateRangeActual(VariableKF::invR00Approx, invR00Approx);
    updateRangeActual(VariableKF::invR11Approx, invR11Approx);
    updateRangeActual(VariableKF::invR00Cor, invR00Cor);
    updateRangeActual(VariableKF::invR11Cor, invR11Cor);
    updateRangeActual(VariableKF::invR00, invR00);
    updateRangeActual(VariableKF::invR11, invR11);
    updateRangeActual(VariableKF::K00, K00);
    updateRangeActual(VariableKF::K10, K10);
    updateRangeActual(VariableKF::K21, K21);
    updateRangeActual(VariableKF::K31, K31);
    // create updated state
    states_.emplace_back(State(state, {x0, x1, x2, x3, 0., C00, C11, C22, C33, C01, C23, 0., 0., 0.}));
    state = &states_.back();
    updateRangeActual(VariableKF::x0, x0);
    updateRangeActual(VariableKF::x1, x1);
    updateRangeActual(VariableKF::x2, x2);
    updateRangeActual(VariableKF::x3, x3);
    updateRangeActual(VariableKF::C00, C00);
    updateRangeActual(VariableKF::C01, C01);
    updateRangeActual(VariableKF::C11, C11);
    updateRangeActual(VariableKF::C22, C22);
    updateRangeActual(VariableKF::C23, C23);
    updateRangeActual(VariableKF::C33, C33);
  }

  // remove and return first element of deque, returns nullptr if empty
  template <class T>
  T* KalmanFilter::pop_front(std::deque<T*>& ts) const {
    T* t = nullptr;
    if (!ts.empty()) {
      t = ts.front();
      ts.pop_front();
    }
    return t;
  }

}  // namespace trklet
