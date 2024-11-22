#include "L1Trigger/TrackerTFP/interface/KalmanFilter.h"

#include <numeric>
#include <algorithm>
#include <iterator>
#include <deque>
#include <vector>
#include <set>
#include <utility>
#include <cmath>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  KalmanFilter::KalmanFilter(const ParameterSet& iConfig,
                             const Setup* setup,
                             const DataFormats* dataFormats,
                             const LayerEncoding* layerEncoding,
                             KalmanFilterFormats* kalmanFilterFormats,
                             vector<TrackKF>& tracks,
                             vector<StubKF>& stubs)
      : enableTruncation_(iConfig.getParameter<bool>("EnableTruncation")),
        setup_(setup),
        dataFormats_(dataFormats),
        layerEncoding_(layerEncoding),
        kalmanFilterFormats_(kalmanFilterFormats),
        tracks_(tracks),
        stubs_(stubs),
        layer_(0) {}

  // fill output products
  void KalmanFilter::produce(const vector<vector<TrackCTB*>>& tracksIn,
                             const vector<vector<Stub*>>& stubsIn,
                             vector<vector<TrackKF*>>& tracksOut,
                             vector<vector<vector<StubKF*>>>& stubsOut,
                             int& numAcceptedStates,
                             int& numLostStates,
                             deque<pair<double, double>>& chi2s) {
    for (int channel = 0; channel < dataFormats_->numChannel(Process::kf); channel++) {
      deque<State*> stream;
      // proto state creation
      createProtoStates(tracksIn, stubsIn, channel, stream);
      // seed building
      for (layer_ = 0; layer_ < setup_->kfMaxSeedingLayer(); layer_++)
        addSeedLayer(stream);
      // calulcate seed parameter
      calcSeeds(stream);
      // Propagate state to each layer in turn, updating it with all viable stub combinations there, using KF maths
      for (layer_ = setup_->kfNumSeedStubs(); layer_ < setup_->numLayers(); layer_++)
        addLayer(stream);
      // count total number of final states
      const int nStates =
          accumulate(stream.begin(), stream.end(), 0, [](int sum, State* state) { return sum += (state ? 1 : 0); });
      // apply truncation
      if (enableTruncation_ && (int)stream.size() > setup_->numFramesHigh())
        stream.resize(setup_->numFramesHigh());
      // cycle event, remove gaps
      stream.erase(remove(stream.begin(), stream.end(), nullptr), stream.end());
      // store number of states which got taken into account
      numAcceptedStates += (int)stream.size();
      // store number of states which got not taken into account due to truncation
      numLostStates += nStates - (int)stream.size();
      // apply final cuts
      vector<Track> finals;
      finals.reserve(stream.size());
      finalize(stream, finals);
      // best track per candidate selection
      vector<Track*> best;
      best.reserve(stream.size());
      accumulator(finals, best);
      // store chi2s
      for (Track* track : best)
        chi2s.emplace_back(track->chi20_, track->chi21_);
      // Transform States into Tracks
      vector<TrackKF*>& tracks = tracksOut[channel];
      vector<vector<StubKF*>>& stubs = stubsOut[channel];
      conv(best, tracks, stubs);
    }
  }

  // create Proto States
  void KalmanFilter::createProtoStates(const std::vector<std::vector<TrackCTB*>>& tracksIn,
                                       const std::vector<std::vector<Stub*>>& stubsIn,
                                       int channel,
                                       deque<State*>& stream) {
    static const int numLayers = setup_->numLayers();
    const int offsetL = channel * numLayers;
    const vector<TrackCTB*>& tracksChannel = tracksIn[channel];
    int trackId(0);
    for (int frame = 0; frame < (int)tracksChannel.size();) {
      TrackCTB* track = tracksChannel[frame];
      if (!track) {
        frame++;
        continue;
      }
      const auto begin = next(tracksChannel.begin(), frame);
      const auto end = find_if(begin + 1, tracksChannel.end(), [](TrackCTB* track) { return track; });
      const int size = distance(begin, end);
      vector<vector<Stub*>> stubs(numLayers);
      for (vector<Stub*>& layer : stubs)
        layer.reserve(size);
      for (int layer = 0; layer < numLayers; layer++) {
        const vector<Stub*>& layerAll = stubsIn[layer + offsetL];
        vector<Stub*>& layerTrack = stubs[layer];
        for (int frameS = 0; frameS < size; frameS++) {
          Stub* stub = layerAll[frameS + frame];
          if (!stub)
            break;
          layerTrack.push_back(stub);
        }
      }
      const TTBV& maybePattern = layerEncoding_->maybePattern(track->zT());
      states_.emplace_back(kalmanFilterFormats_, track, stubs, maybePattern, trackId++);
      stream.insert(stream.end(), size - 1, nullptr);
      stream.push_back(&states_.back());
      frame += size;
    }
  }

  // calulcate seed parameter
  void KalmanFilter::calcSeeds(deque<State*>& stream) {
    auto update = [this](State* s) {
      updateRangeActual(VariableKF::m0, s->m0());
      updateRangeActual(VariableKF::m1, s->m1());
      updateRangeActual(VariableKF::v0, s->v0());
      updateRangeActual(VariableKF::v1, s->v1());
      updateRangeActual(VariableKF::H00, s->H00());
      updateRangeActual(VariableKF::H12, s->H12());
    };
    for (State*& state : stream) {
      if (!state)
        continue;
      State* s1 = state->parent();
      State* s0 = s1->parent();
      update(s0);
      update(s1);
      const double dH = digi(VariableKF::dH, s1->H00() - s0->H00());
      const double invdH = digi(VariableKF::invdH, 1.0 / dH);
      const double invdH2 = digi(VariableKF::invdH2, 1.0 / dH / dH);
      const double H12 = digi(VariableKF::H2, s1->H00() * s1->H00());
      const double H02 = digi(VariableKF::H2, s0->H00() * s0->H00());
      const double H32 = digi(VariableKF::H2, s1->H12() * s1->H12());
      const double H22 = digi(VariableKF::H2, s0->H12() * s0->H12());
      const double H1m0 = digi(VariableKF::Hm0, s1->H00() * s0->m0());
      const double H0m1 = digi(VariableKF::Hm0, s0->H00() * s1->m0());
      const double H3m2 = digi(VariableKF::Hm1, s1->H12() * s0->m1());
      const double H2m3 = digi(VariableKF::Hm1, s0->H12() * s1->m1());
      const double H1v0 = digi(VariableKF::Hv0, s1->H00() * s0->v0());
      const double H0v1 = digi(VariableKF::Hv0, s0->H00() * s1->v0());
      const double H3v2 = digi(VariableKF::Hv1, s1->H12() * s0->v1());
      const double H2v3 = digi(VariableKF::Hv1, s0->H12() * s1->v1());
      const double H12v0 = digi(VariableKF::H2v0, H12 * s0->v0());
      const double H02v1 = digi(VariableKF::H2v0, H02 * s1->v0());
      const double H32v2 = digi(VariableKF::H2v1, H32 * s0->v1());
      const double H22v3 = digi(VariableKF::H2v1, H22 * s1->v1());
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
      // cut on eta sector boundaries
      /*const bool invalidX3 = abs(x3) > zT.base() / 2.;
      // cut on triple found inv2R window
      const bool invalidX0 = abs(x0) > 1.5 * inv2R.base();
      // cut on triple found phiT window
      const bool invalidX1 = abs(x1) > 1.5 * phiT.base();
      // cot cut
      const bool invalidX2 = abs(x2) > maxCot;
      if (invalidX3 || invalidX0 || invalidX1 || invalidX2) {
        state = nullptr;
        continue;
      }*/
      // create updated state
      static const double chi20 = digi(VariableKF::chi20, 0.);
      static const double chi21 = digi(VariableKF::chi21, 0.);
      states_.emplace_back(State(s1, {x0, x1, x2, x3, chi20, chi21, C00, C11, C22, C33, C01, C23}));
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
  }

  // apply final cuts
  void KalmanFilter::finalize(const deque<State*>& stream, vector<Track>& finals) {
    for (State* state : stream) {
      TrackCTB* track = state->track();
      int numConsistent(0);
      int numConsistentPS(0);
      TTBV hitPattern = state->hitPattern();
      vector<StubCTB*> stubs;
      vector<double> phis;
      vector<double> zs;
      stubs.reserve(setup_->numLayers());
      phis.reserve(setup_->numLayers());
      zs.reserve(setup_->numLayers());
      double chi20(0.);
      double chi21(0.);
      // stub residual cut
      State* s = state;
      while ((s = s->parent())) {
        const double dPhi = state->x1() + s->H00() * state->x0();
        const double dZ = state->x3() + s->H12() * state->x2();
        const double phi = digi(VariableKF::m0, s->m0() - dPhi);
        const double z = digi(VariableKF::m1, s->m1() - dZ);
        const bool validPhi = dataFormats_->format(Variable::phi, Process::kf).inRange(phi);
        const bool validZ = dataFormats_->format(Variable::z, Process::kf).inRange(z);
        StubCTB& stubCTB = s->stub()->stubCTB_;
        if (validPhi && validZ) {
          chi20 += pow(phi, 2);
          chi21 += pow(z, 2);
          stubs.push_back(&stubCTB);
          phis.push_back(phi);
          zs.push_back(z);
          if (abs(phi) <= s->dPhi() && abs(z) <= s->dZ()) {
            numConsistent++;
            if (setup_->psModule(stubCTB.frame().first))
              numConsistentPS++;
          }
        } else
          hitPattern.reset(s->layer());
      }
      const double ndof = hitPattern.count() - 2;
      chi20 /= ndof;
      chi21 /= ndof;
      // layer cut
      bool invalidLayers = hitPattern.count() < setup_->kfMinLayers();
      // track parameter cuts
      const double cotTrack = dataFormats_->format(Variable::cot, Process::kf).digi(track->zT() / setup_->chosenRofZ());
      const double inv2R = state->x0() + track->inv2R();
      const double phiT = state->x1() + track->phiT();
      const double cot = state->x2() + cotTrack;
      const double zT = state->x3() + track->zT();
      // pt cut
      const bool validX0 = dataFormats_->format(Variable::inv2R, Process::kf).inRange(inv2R);
      // cut on phi sector boundaries
      const bool validX1 = dataFormats_->format(Variable::phiT, Process::kf).inRange(phiT);
      // cot cut
      const bool validX2 = dataFormats_->format(Variable::cot, Process::kf).inRange(cot);
      // zT cut
      const bool validX3 = dataFormats_->format(Variable::zT, Process::kf).inRange(zT);
      if (invalidLayers || !validX0 || !validX1 || !validX2 || !validX3)
        continue;
      const int trackId = state->trackId();
      finals_.emplace_back(trackId, numConsistent, numConsistentPS, inv2R, phiT, cot, zT, chi20, chi21, hitPattern, track, stubs, phis, zs);
    }
  }

  // Transform States into Tracks
  void KalmanFilter::conv(const vector<Track*>& best, vector<TrackKF*>& tracks, vector<vector<StubKF*>>& stubs) {
    static const DataFormat& dfInv2R = dataFormats_->format(Variable::inv2R, Process::ht);
    static const DataFormat& dfPhiT = dataFormats_->format(Variable::phiT, Process::ht);
    tracks.reserve(best.size());
    for (vector<StubKF*>& layer : stubs)
      layer.reserve(best.size());
    for (Track* track : best) {
      const vector<int> layers = track->hitPattern_.ids();
      for (int iStub = 0; iStub < track->hitPattern_.count(); iStub++) {
        StubCTB* s = track->stubs_[iStub];
        stubs_.emplace_back(*s, s->r(), track->phi_[iStub], track->z_[iStub], s->dPhi(), s->dZ());
        stubs[layers[iStub]].push_back(&stubs_.back());
      }
      TrackCTB* trackCTB = track->track_;
      const bool inInv2R = dfInv2R.integer(track->inv2R_) == dfInv2R.integer(trackCTB->inv2R());
      const bool inPhiT = dfPhiT.integer(track->phiT_) == dfPhiT.integer(trackCTB->phiT());
      const TTBV match(inInv2R && inPhiT, 1);
      tracks_.emplace_back(*trackCTB, track->inv2R_, track->phiT_, track->cot_, track->zT_, match);
      tracks.push_back(&tracks_.back());
    }
  }

  // adds a layer to states
  void KalmanFilter::addLayer(deque<State*>& stream) {
    // Latency of KF Associator block firmware
    static constexpr int latency = 5;
    // dynamic state container for clock accurate emulation
    deque<State*> streamOutput;
    // Memory stack used to handle combinatorics
    deque<State*> stack;
    // static delay container
    deque<State*> delay(latency, nullptr);
    // each trip corresponds to a f/w clock tick
    // done if no states to process left, taking as much time as needed
    while (!stream.empty() || !stack.empty() ||
           !all_of(delay.begin(), delay.end(), [](const State* state) { return state == nullptr; })) {
      State* state = pop_front(stream);
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
    stream = streamOutput;
    // Update state with next stub using KF maths
    for (State*& state : stream)
      if (state && state->hitPattern().pmEncode() == layer_)
        update(state);
  }

  // adds a layer to states to build seeds
  void KalmanFilter::addSeedLayer(deque<State*>& stream) {
    // Latency of KF Associator block firmware
    static constexpr int latency = 5;
    // dynamic state container for clock accurate emulation
    deque<State*> streamOutput;
    // Memory stack used to handle combinatorics
    deque<State*> stack;
    // static delay container
    deque<State*> delay(latency, nullptr);
    // each trip corresponds to a f/w clock tick
    // done if no states to process left, taking as much time as needed
    while (!stream.empty() || !stack.empty() ||
           !all_of(delay.begin(), delay.end(), [](const State* state) { return state == nullptr; })) {
      State* state = pop_front(stream);
      // Process a combinatoric state if no (non-combinatoric?) state available
      if (!state)
        state = pop_front(stack);
      streamOutput.push_back(state);
      // The remainder of the code in this loop deals with combinatoric states.
      if (state)
        state = state->combSeed(states_, layer_);
      delay.push_back(state);
      state = pop_front(delay);
      if (state)
        stack.push_back(state);
    }
    stream = streamOutput;
    // Update state with next stub using KF maths
    for (State*& state : stream)
      if (state)
        state = state->update(states_, layer_);
  }

  // best state selection
  void KalmanFilter::accumulator(vector<Track>& finals, vector<Track*>& best) {
    // create container of pointer to make sorts less CPU intense
    //for (Track& track : finals)
      //best.push_back(&track);
    transform(finals.begin(), finals.end(), back_inserter(best), [](Track& track) { return &track; });
    // prepare arrival order
    vector<int> trackIds;
    trackIds.reserve(best.size());
    for (Track* track : best) {
      const int trackId = track->trackId_;
      if (find_if(trackIds.begin(), trackIds.end(), [trackId](int id) { return id == trackId; }) == trackIds.end())
        trackIds.push_back(trackId);
    }
    // sort in chi2
    auto smallerChi2 = [](Track* lhs, Track* rhs) { return lhs->chi20_ + lhs->chi21_ < rhs->chi20_ + rhs->chi21_; };
    stable_sort(best.begin(), best.end(), smallerChi2);
    // sort in number of consistent stubs
    auto moreConsistentLayers = [](Track* lhs, Track* rhs) { return lhs->numConsistent_ > rhs->numConsistent_; };
    stable_sort(best.begin(), best.end(), moreConsistentLayers);
    // sort in number of consistent ps stubs
    auto moreConsistentLayersPS = [](Track* lhs, Track* rhs) { return lhs->numConsistentPS_ > rhs->numConsistentPS_; };
    stable_sort(best.begin(), best.end(), moreConsistentLayersPS);
    // sort in track id as arrived
    auto order = [&trackIds](auto lhs, auto rhs) {
      const auto l = find(trackIds.begin(), trackIds.end(), lhs->trackId_);
      const auto r = find(trackIds.begin(), trackIds.end(), rhs->trackId_);
      return distance(r, l) < 0;
    };
    stable_sort(best.begin(), best.end(), order);
    // keep first state (best due to previous sorts) per track id
    auto same = [](Track* lhs, Track* rhs) { return lhs->trackId_ == rhs->trackId_; };
    best.erase(unique(best.begin(), best.end(), same), best.end());
  }

  // updates state
  void KalmanFilter::update(State*& state) {
    // All variable names & equations come from Fruhwirth KF paper http://dx.doi.org/10.1016/0168-9002%2887%2990887-4", where F taken as unit matrix. Stub uncertainties projected onto (phi,z), assuming no correlations between r-phi & r-z planes.
    // stub phi residual wrt input helix
    const double m0 = state->m0();
    // stub z residual wrt input helix
    const double m1 = state->m1();
    // stub projected phi uncertainty squared);
    const double v0 = state->v0();
    // stub projected z uncertainty squared
    const double v1 = state->v1();
    // Derivative of predicted stub coords wrt helix params: stub radius minus chosenRofPhi
    const double H00 = state->H00();
    // Derivative of predicted stub coords wrt helix params: stub radius minus chosenRofZ
    const double H12 = state->H12();
    updateRangeActual(VariableKF::m0, m0);
    updateRangeActual(VariableKF::m1, m1);
    updateRangeActual(VariableKF::v0, v0);
    updateRangeActual(VariableKF::v1, v1);
    updateRangeActual(VariableKF::H00, H00);
    updateRangeActual(VariableKF::H12, H12);
    // helix inv2R wrt input helix
    double x0 = state->x0();
    // helix phi at radius ChosenRofPhi wrt input helix
    double x1 = state->x1();
    // helix cot(Theta) wrt input helix
    double x2 = state->x2();
    // helix z at radius chosenRofZ wrt input helix
    double x3 = state->x3();
    // cov. matrix
    double C00 = state->C00();
    double C01 = state->C01();
    double C11 = state->C11();
    double C22 = state->C22();
    double C23 = state->C23();
    double C33 = state->C33();
    // chi2s
    double chi20 = state->chi20();
    double chi21 = state->chi21();
    // stub phi residual wrt current state
    const double r0C = digi(VariableKF::x1, m0 - x1);
    const double r0 = digi(VariableKF::r0, r0C - x0 * H00);
    // stub z residual wrt current state
    const double r1C = digi(VariableKF::x3, m1 - x3);
    const double r1 = digi(VariableKF::r1, r1C - x2 * H12);
    // matrix S = H*C
    const double S00 = digi(VariableKF::S00, C01 + H00 * C00);
    const double S01 = digi(VariableKF::S01, C11 + H00 * C01);
    const double S12 = digi(VariableKF::S12, C23 + H12 * C22);
    const double S13 = digi(VariableKF::S13, C33 + H12 * C23);
    // Cov. matrix of predicted residuals R = V+HCHt = C+H*St
    const double R00 = digi(VariableKF::R00, v0 + S01 + H00 * S00);
    const double R11 = digi(VariableKF::R11, v1 + S13 + H12 * S12);
    // improved dynamic cancelling
    const int msb0 = max(0, (int)ceil(log2(R00 / base(VariableKF::R00))));
    const int msb1 = max(0, (int)ceil(log2(R11 / base(VariableKF::R11))));
    const int shift0 = width(VariableKF::R00) - msb0;
    const int shift1 = width(VariableKF::R11) - msb1;
    const double R00Shifted = R00 * pow(2., shift0);
    const double R11Shifted = R11 * pow(2., shift1);
    const double R00Rough = digi(VariableKF::R00Rough, R00Shifted);
    const double R11Rough = digi(VariableKF::R11Rough, R11Shifted);
    const double invR00Approx = digi(VariableKF::invR00Approx, 1. / R00Rough);
    const double invR11Approx = digi(VariableKF::invR11Approx, 1. / R11Rough);
    const double invR00Cor = digi(VariableKF::invR00Cor, 2. - invR00Approx * R00Shifted);
    const double invR11Cor = digi(VariableKF::invR11Cor, 2. - invR11Approx * R11Shifted);
    const double invR00 = digi(VariableKF::invR00, invR00Approx * invR00Cor);
    const double invR11 = digi(VariableKF::invR11, invR11Approx * invR11Cor);
    // shift S to "undo" shifting of R
    auto digiShifted = [](double val, double base) { return floor(val / base * 2. + 1.e-11) * base / 2.; };
    const double S00Shifted = digiShifted(S00 * pow(2., shift0), base(VariableKF::S00Shifted));
    const double S01Shifted = digiShifted(S01 * pow(2., shift0), base(VariableKF::S01Shifted));
    const double S12Shifted = digiShifted(S12 * pow(2., shift1), base(VariableKF::S12Shifted));
    const double S13Shifted = digiShifted(S13 * pow(2., shift1), base(VariableKF::S13Shifted));
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
    // squared residuals
    const double r0Shifted = digiShifted(r0 * pow(2., shift0), base(VariableKF::r0Shifted));
    const double r1Shifted = digiShifted(r1 * pow(2., shift1), base(VariableKF::r1Shifted));
    const double r02 = digi(VariableKF::r02, r0 * r0);
    const double r12 = digi(VariableKF::r12, r1 * r1);
    chi20 = digi(VariableKF::chi20, chi20 + r02 * invR00);
    chi21 = digi(VariableKF::chi21, chi21 + r12 * invR11);
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
    updateRangeActual(VariableKF::r0Shifted, r0Shifted);
    updateRangeActual(VariableKF::r1Shifted, r1Shifted);
    updateRangeActual(VariableKF::r02, r02);
    updateRangeActual(VariableKF::r12, r12);
    // range checks
    const bool validX0 = inRange(VariableKF::x0, x0);
    const bool validX1 = inRange(VariableKF::x1, x1);
    const bool validX2 = inRange(VariableKF::x2, x2);
    const bool validX3 = inRange(VariableKF::x3, x3);
    // chi2 cut
    const double dof = state->hitPattern().count() - 1;
    const double chi2 = (chi20 + chi21) / 2.;
    const bool validChi2 = chi2 < setup_->kfCutChi2() * dof;
    if (!validX0 || !validX1 || !validX2 || !validX3 || !validChi2) {
      state = nullptr;
      return;
    }
    // create updated state
    states_.emplace_back(State(state, {x0, x1, x2, x3, chi20, chi21, C00, C11, C22, C33, C01, C23}));
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
    updateRangeActual(VariableKF::chi20, chi20);
    updateRangeActual(VariableKF::chi21, chi21);
  }

  // remove and return first element of deque, returns nullptr if empty
  template <class T>
  T* KalmanFilter::pop_front(deque<T*>& ts) const {
    T* t = nullptr;
    if (!ts.empty()) {
      t = ts.front();
      ts.pop_front();
    }
    return t;
  }

}  // namespace trackerTFP
