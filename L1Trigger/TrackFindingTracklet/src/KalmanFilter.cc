#include "L1Trigger/TrackFindingTracklet/interface/KalmanFilter.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"

#include <numeric>
#include <algorithm>
#include <iterator>
#include <deque>
#include <vector>
#include <set>
#include <cmath>

using namespace std;
using namespace edm;
using namespace tt;
using namespace tmtt;

namespace trklet {

  KalmanFilter::KalmanFilter(const ParameterSet& iConfig,
                             const Setup* setup,
                             const DataFormats* dataFormats,
                             KalmanFilterFormats* kalmanFilterFormats,
                             Settings* settings,
                             KFParamsComb* tmtt,
                             int region,
                             TTTracks& ttTracks)
      : enableTruncation_(iConfig.getParameter<bool>("EnableTruncation")),
        use5ParameterFit_(iConfig.getParameter<bool>("Use5ParameterFit")),
        useSimmulation_(iConfig.getParameter<bool>("UseKFsimmulation")),
        useTTStubResiduals_(iConfig.getParameter<bool>("UseTTStubResiduals")),
        setup_(setup),
        dataFormats_(dataFormats),
        kalmanFilterFormats_(kalmanFilterFormats),
        settings_(settings),
        tmtt_(tmtt),
        region_(region),
        ttTracks_(ttTracks),
        layer_(0) {}

  // read in and organize input tracks and stubs
  void KalmanFilter::consume(const StreamsTrack& streamsTrack, const StreamsStub& streamsStub) {
    static const int numLayers = setup_->numLayers();
    const int offset = region_ * numLayers;
    const StreamTrack& streamTrack = streamsTrack[region_];
    const int numTracks = accumulate(streamTrack.begin(), streamTrack.end(), 0, [](int sum, const FrameTrack& f) {
      return sum += (f.first.isNull() ? 0 : 1);
    });
    int numStubs(0);
    for (int layer = 0; layer < numLayers; layer++) {
      const StreamStub& streamStub = streamsStub[offset + layer];
      numStubs += accumulate(streamStub.begin(), streamStub.end(), 0, [](int sum, const FrameStub& f) {
        return sum += (f.first.isNull() ? 0 : 1);
      });
    }
    tracks_.reserve(numTracks);
    stubs_.reserve(numStubs);
    int trackId(0);
    for (int frame = 0; frame < (int)streamTrack.size(); frame++) {
      const FrameTrack& frameTrack = streamTrack[frame];
      if (frameTrack.first.isNull()) {
        stream_.push_back(nullptr);
        continue;
      }
      tracks_.emplace_back(frameTrack, dataFormats_);
      TrackDR* track = &tracks_.back();
      vector<Stub*> stubs(numLayers, nullptr);
      TTBV hitPattern(0, setup_->numLayers());
      for (int layer = 0; layer < numLayers; layer++) {
        const FrameStub& frameStub = streamsStub[offset + layer][frame];
        if (frameStub.first.isNull())
          continue;
        stubs_.emplace_back(kalmanFilterFormats_, frameStub);
        stubs[layer] = &stubs_.back();
        hitPattern.set(layer);
      }
      if (hitPattern.count(0, setup_->kfMaxSeedingLayer()) < setup_->kfNumSeedStubs()) {
        stream_.push_back(nullptr);
        continue;
      }
      states_.emplace_back(kalmanFilterFormats_, track, stubs, trackId++);
      stream_.push_back(&states_.back());
      if (enableTruncation_ && trackId == setup_->kfMaxTracks())
        break;
    }
  }

  // call old KF
  void KalmanFilter::simulate(tt::StreamsStub& streamsStub, tt::StreamsTrack& streamsTrack) {
    static vector<double> zTs;
    if (zTs.empty()) {
      zTs.reserve(settings_->etaRegions().size());
      for (double eta : settings_->etaRegions())
        zTs.emplace_back(sinh(eta) * settings_->chosenRofZ());
    }
    finals_.reserve(states_.size());
    for (const State& state : states_) {
      TrackDR* trackFound = state.track();
      const TTTrackRef& ttTrackRef = trackFound->frame().first;
      const double qOverPt = -trackFound->inv2R() / setup_->invPtToDphi();
      const double phi0 =
          deltaPhi(trackFound->phiT() - setup_->chosenRofPhi() * trackFound->inv2R() + region_ * setup_->baseRegion());
      const double tanLambda = trackFound->zT() / setup_->chosenRofZ();
      static constexpr double z0 = 0;
      static constexpr double helixD0 = 0.;
      vector<tmtt::Stub> stubs;
      vector<tmtt::Stub*> stubsFound;
      stubs.reserve(state.trackPattern().count());
      stubsFound.reserve(state.trackPattern().count());
      const vector<Stub*>& stubsState = state.stubs();
      for (int layer = 0; layer < setup_->numLayers(); layer++) {
        if (!stubsState[layer])
          continue;
        const StubDR& stub = stubsState[layer]->stubDR_;
        const TTStubRef& ttStubRef = stub.frame().first;
        SensorModule* sensorModule = setup_->sensorModule(ttStubRef);
        double r, phi, z;
        if (useTTStubResiduals_) {
          const GlobalPoint gp = setup_->stubPos(ttStubRef);
          r = gp.perp();
          phi = gp.phi();
          z = gp.z();
        } else {
          r = stub.r() + setup_->chosenRofPhi();
          phi = deltaPhi(stub.phi() + trackFound->phiT() + stub.r() * trackFound->inv2R() +
                         region_ * setup_->baseRegion());
          z = stub.z() + trackFound->zT() + (r - setup_->chosenRofZ()) * tanLambda;
        }
        int layerId = setup_->layerId(ttStubRef);
        if (layerId > 10 && z < 0.)
          layerId += 10;
        int layerIdReduced = setup_->layerId(ttStubRef);
        if (layerIdReduced == 6)
          layerIdReduced = 11;
        else if (layerIdReduced == 5)
          layerIdReduced = 12;
        else if (layerIdReduced == 4)
          layerIdReduced = 13;
        else if (layerIdReduced == 3)
          layerIdReduced = 15;
        if (layerIdReduced > 10)
          layerIdReduced -= 8;
        const double stripPitch = sensorModule->pitchRow();
        const double stripLength = sensorModule->pitchCol();
        const bool psModule = sensorModule->psModule();
        const bool barrel = sensorModule->barrel();
        const bool tiltedBarrel = sensorModule->tilted();
        stubs.emplace_back(
            ttStubRef, r, phi, z, layerId, layerIdReduced, stripPitch, stripLength, psModule, barrel, tiltedBarrel);
        stubsFound.push_back(&stubs.back());
      }
      const int iPhiSec = region_;
      const double zTtrack = ttTrackRef->z0() + settings_->chosenRofZ() * ttTrackRef->tanL();
      int iEtaReg = 0;
      for (; iEtaReg < 15; iEtaReg++)
        if (zTtrack < zTs[iEtaReg + 1])
          break;
      const L1track3D l1track3D(settings_, stubsFound, qOverPt, phi0, z0, tanLambda, helixD0, iPhiSec, iEtaReg);
      const L1fittedTrack trackFitted(tmtt_->fit(l1track3D));
      if (!trackFitted.accepted())
        continue;
      static constexpr int trackId = 0;
      static constexpr int numConsistent = 0;
      static constexpr int numConsistentPS = 0;
      const double inv2R = -trackFitted.qOverPt() * setup_->invPtToDphi();
      const double phiT =
          deltaPhi(trackFitted.phi0() + inv2R * setup_->chosenRofPhi() - region_ * setup_->baseRegion());
      const double cot = trackFitted.tanLambda();
      const double zT = trackFitted.z0() + cot * setup_->chosenRofZ();
      if (!dataFormats_->format(Variable::inv2R, Process::kf).inRange(inv2R, true))
        continue;
      if (!dataFormats_->format(Variable::phiT, Process::kf).inRange(phiT, true))
        continue;
      if (!dataFormats_->format(Variable::cot, Process::kf).inRange(cot, true))
        continue;
      if (!dataFormats_->format(Variable::zT, Process::kf).inRange(zT, true))
        continue;
      const double d0 = trackFitted.d0();
      const double x0 = inv2R - trackFound->inv2R();
      const double x1 = phiT - trackFound->phiT();
      const double x2 = cot - tanLambda;
      const double x3 = zT - trackFound->zT();
      const double x4 = d0;
      TTBV hitPattern(0, setup_->numLayers());
      vector<StubKF> stubsKF;
      stubsKF.reserve(setup_->numLayers());
      for (tmtt::Stub* stub : trackFitted.stubs()) {
        if (!stub)
          continue;
        const auto it = find_if(stubsState.begin(), stubsState.end(), [stub](Stub* state) {
          return state && (stub->ttStubRef() == state->stubDR_.frame().first);
        });
        const StubDR& s = (*it)->stubDR_;
        const double r = s.r();
        const double r0 = r + setup_->chosenRofPhi();
        const double phi = s.phi() - (x1 + r * x0 + x4 / r0);
        const double z = s.z() - (x3 + (r0 - setup_->chosenRofZ()) * x2);
        const double dPhi = s.dPhi();
        const double dZ = s.dZ();
        const int layer = distance(stubsState.begin(), it);
        if (!dataFormats_->format(Variable::phi, Process::kf).inRange(phi, true))
          continue;
        if (!dataFormats_->format(Variable::z, Process::kf).inRange(z, true))
          continue;
        hitPattern.set(layer);
        stubsKF.emplace_back(s, r, phi, z, dPhi, dZ);
      }
      if (hitPattern.count() < setup_->kfMinLayers())
        continue;
      const TrackKF trackKF(*trackFound, inv2R, phiT, cot, zT);
      finals_.emplace_back(trackId, numConsistent, numConsistentPS, d0, hitPattern, trackKF, stubsKF);
    }
    conv(streamsStub, streamsTrack);
  }

  // fill output products
  void KalmanFilter::produce(StreamsStub& streamsStub,
                             StreamsTrack& streamsTrack,
                             int& numAcceptedStates,
                             int& numLostStates) {
    if (useSimmulation_)
      return simulate(streamsStub, streamsTrack);
    // 5 parameter fit simulation
    if (use5ParameterFit_) {
      // Propagate state to each layer in turn, updating it with all viable stub combinations there, using KF maths
      for (layer_ = 0; layer_ < setup_->numLayers(); layer_++)
        addLayer();
    } else {  // 4 parameter fit emulation
      // seed building
      for (layer_ = 0; layer_ < setup_->kfMaxSeedingLayer(); layer_++)
        addSeedLayer();
      // calulcate seed parameter
      calcSeeds();
      // Propagate state to each layer in turn, updating it with all viable stub combinations there, using KF maths
      for (layer_ = setup_->kfNumSeedStubs(); layer_ < setup_->numLayers(); layer_++)
        addLayer();
    }
    // count total number of final states
    const int nStates =
        accumulate(stream_.begin(), stream_.end(), 0, [](int sum, State* state) { return sum += (state ? 1 : 0); });
    // apply truncation
    if (enableTruncation_ && (int)stream_.size() > setup_->numFramesHigh())
      stream_.resize(setup_->numFramesHigh());
    // cycle event, remove gaps
    stream_.erase(remove(stream_.begin(), stream_.end(), nullptr), stream_.end());
    // store number of states which got taken into account
    numAcceptedStates += (int)stream_.size();
    // store number of states which got not taken into account due to truncation
    numLostStates += nStates - (int)stream_.size();
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
      TTBV hitPattern = state->hitPattern();
      vector<StubKF> stubsKF;
      stubsKF.reserve(setup_->numLayers());
      // stub residual cut
      State* s = state;
      while ((s = s->parent())) {
        const double dPhi = state->x1() + s->H00() * state->x0() + state->x4() / s->H04();
        const double dZ = state->x3() + s->H12() * state->x2();
        const double phi = digi(VariableKF::m0, s->m0() - dPhi);
        const double z = digi(VariableKF::m1, s->m1() - dZ);
        const bool validPhi = dataFormats_->format(Variable::phi, Process::kf).inRange(phi);
        const bool validZ = dataFormats_->format(Variable::z, Process::kf).inRange(z);
        if (validPhi && validZ) {
          const double r = s->H00();
          const double dPhi = s->d0();
          const double dZ = s->d1();
          const StubDR& stubDR = s->stub()->stubDR_;
          stubsKF.emplace_back(stubDR, r, phi, z, dPhi, dZ);
          if (abs(phi) <= dPhi && abs(z) <= dZ) {
            numConsistent++;
            if (setup_->psModule(stubDR.frame().first))
              numConsistentPS++;
          }
        } else
          hitPattern.reset(s->layer());
      }
      reverse(stubsKF.begin(), stubsKF.end());
      // layer cut
      bool validLayers = hitPattern.count() >= setup_->kfMinLayers();
      // track parameter cuts
      const double cotTrack =
          dataFormats_->format(Variable::cot, Process::kf).digi(state->track()->zT() / setup_->chosenRofZ());
      const double inv2R = state->x0() + state->track()->inv2R();
      const double phiT = state->x1() + state->track()->phiT();
      const double cot = state->x2() + cotTrack;
      const double zT = state->x3() + state->track()->zT();
      const double d0 = state->x4();
      // pt cut
      const bool validX0 = dataFormats_->format(Variable::inv2R, Process::kf).inRange(inv2R);
      // cut on phi sector boundaries
      const bool validX1 = abs(phiT) < setup_->baseRegion() / 2.;
      // cot cut
      const bool validX2 = dataFormats_->format(Variable::cot, Process::kf).inRange(cot);
      // zT cut
      const bool validX3 = dataFormats_->format(Variable::zT, Process::kf).inRange(zT);
      if (!validLayers || !validX0 || !validX1 || !validX2 || !validX3)
        continue;
      const int trackId = state->trackId();
      const TrackKF trackKF(*state->track(), inv2R, phiT, cot, zT);
      finals_.emplace_back(trackId, numConsistent, numConsistentPS, d0, hitPattern, trackKF, stubsKF);
    }
  }

  // best state selection
  void KalmanFilter::accumulator() {
    // create container of pointer to make sorts less CPU intense
    vector<Track*> finals;
    finals.reserve(finals_.size());
    transform(finals_.begin(), finals_.end(), back_inserter(finals), [](Track& track) { return &track; });
    // prepare arrival order
    vector<int> trackIds;
    trackIds.reserve(tracks_.size());
    for (Track* track : finals) {
      const int trackId = track->trackId_;
      if (find_if(trackIds.begin(), trackIds.end(), [trackId](int id) { return id == trackId; }) == trackIds.end())
        trackIds.push_back(trackId);
    }
    // sort in number of consistent stubs
    auto moreConsistentLayers = [](Track* lhs, Track* rhs) { return lhs->numConsistent_ > rhs->numConsistent_; };
    stable_sort(finals.begin(), finals.end(), moreConsistentLayers);
    // sort in number of consistent ps stubs
    auto moreConsistentLayersPS = [](Track* lhs, Track* rhs) { return lhs->numConsistentPS_ > rhs->numConsistentPS_; };
    stable_sort(finals.begin(), finals.end(), moreConsistentLayersPS);
    // sort in track id as arrived
    auto order = [&trackIds](auto lhs, auto rhs) {
      const auto l = find(trackIds.begin(), trackIds.end(), lhs->trackId_);
      const auto r = find(trackIds.begin(), trackIds.end(), rhs->trackId_);
      return distance(r, l) < 0;
    };
    stable_sort(finals.begin(), finals.end(), order);
    // keep first state (best due to previous sorts) per track id
    const auto it =
        unique(finals.begin(), finals.end(), [](Track* lhs, Track* rhs) { return lhs->trackId_ == rhs->trackId_; });
    finals.erase(it, finals.end());
    // apply to actual track container
    int i(0);
    for (Track* track : finals)
      finals_[i++] = *track;
    finals_.resize(i);
  }

  // Transform States into output products
  void KalmanFilter::conv(StreamsStub& streamsStub, StreamsTrack& streamsTrack) {
    const int offset = region_ * setup_->numLayers();
    StreamTrack& streamTrack = streamsTrack[region_];
    streamTrack.reserve(stream_.size());
    for (int layer = 0; layer < setup_->numLayers(); layer++)
      streamsStub[offset + layer].reserve(stream_.size());
    for (const Track& track : finals_) {
      streamTrack.emplace_back(track.trackKF_.frame());
      const TTBV& hitPattern = track.hitPattern_;
      const vector<StubKF>& stubsKF = track.stubsKF_;
      int i(0);
      for (int layer = 0; layer < setup_->numLayers(); layer++)
        streamsStub[offset + layer].emplace_back(hitPattern.test(layer) ? stubsKF[i++].frame() : FrameStub());
      // store d0 in copied TTTracks
      if (use5ParameterFit_) {
        const TTTrackRef& ttTrackRef = track.trackKF_.frame().first;
        ttTracks_.emplace_back(ttTrackRef->rInv(),
                               ttTrackRef->phi(),
                               ttTrackRef->tanL(),
                               ttTrackRef->z0(),
                               track.d0_,
                               ttTrackRef->chi2XY(),
                               ttTrackRef->chi2Z(),
                               ttTrackRef->trkMVA1(),
                               ttTrackRef->trkMVA2(),
                               ttTrackRef->trkMVA3(),
                               ttTrackRef->hitPattern(),
                               5,
                               setup_->bField());
        ttTracks_.back().setPhiSector(ttTrackRef->phiSector());
        ttTracks_.back().setEtaSector(ttTrackRef->etaSector());
        ttTracks_.back().setTrackSeedType(ttTrackRef->trackSeedType());
        ttTracks_.back().setStubPtConsistency(ttTrackRef->stubPtConsistency());
        ttTracks_.back().setStubRefs(ttTrackRef->getStubRefs());
      }
    }
  }

  // calculates the helix params & their cov. matrix from a pair of stubs
  void KalmanFilter::calcSeeds() {
    auto update = [this](State* s) {
      updateRangeActual(VariableKF::m0, s->m0());
      updateRangeActual(VariableKF::m1, s->m1());
      updateRangeActual(VariableKF::v0, s->v0());
      updateRangeActual(VariableKF::v1, s->v1());
      updateRangeActual(VariableKF::H00, s->H00());
      updateRangeActual(VariableKF::H12, s->H12());
    };
    for (State*& state : stream_) {
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
      // create updated state
      states_.emplace_back(State(s1, {x0, x1, x2, x3, 0., C00, C11, C22, C33, C01, C23, 0., 0., 0.}));
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

  // adds a layer to states to build seeds
  void KalmanFilter::addSeedLayer() {
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
    while (!stream_.empty() || !stack.empty() ||
           !all_of(delay.begin(), delay.end(), [](const State* state) { return state == nullptr; })) {
      State* state = pop_front(stream_);
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
    stream_ = streamOutput;
    // Update state with next stub using KF maths
    for (State*& state : stream_)
      if (state)
        state = state->update(states_, layer_);
  }

  // adds a layer to states
  void KalmanFilter::addLayer() {
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
    while (!stream_.empty() || !stack.empty() ||
           !all_of(delay.begin(), delay.end(), [](const State* state) { return state == nullptr; })) {
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
      if (state && state->hitPattern().pmEncode() == layer_)
        update(state);
  }

  // updates state
  void KalmanFilter::update4(State*& state) {
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

  // updates state
  void KalmanFilter::update5(State*& state) {
    const double m0 = state->m0();
    const double m1 = state->m1();
    const double v0 = state->v0();
    const double v1 = state->v1();
    const double H00 = state->H00();
    const double H12 = state->H12();
    const double H04 = state->H04();
    double x0 = state->x0();
    double x1 = state->x1();
    double x2 = state->x2();
    double x3 = state->x3();
    double x4 = state->x4();
    double C00 = state->C00();
    double C01 = state->C01();
    double C11 = state->C11();
    double C22 = state->C22();
    double C23 = state->C23();
    double C33 = state->C33();
    double C44 = state->C44();
    double C40 = state->C40();
    double C41 = state->C41();
    const double r0 = m0 - x1 - x0 * H00 - x4 / H04;
    const double r1 = m1 - x3 - x2 * H12;
    const double S00 = C01 + H00 * C00 + C40 / H04;
    const double S01 = C11 + H00 * C01 + C41 / H04;
    const double S12 = C23 + H12 * C22;
    const double S13 = C33 + H12 * C23;
    const double S04 = C41 + H00 * C40 + C44 / H04;
    const double R00 = v0 + S01 + H00 * S00 + S04 / H04;
    const double R11 = v1 + S13 + H12 * S12;
    const double K00 = S00 / R00;
    const double K10 = S01 / R00;
    const double K21 = S12 / R11;
    const double K31 = S13 / R11;
    const double K40 = S04 / R00;
    x0 += r0 * K00;
    x1 += r0 * K10;
    x2 += r1 * K21;
    x3 += r1 * K31;
    x4 += r0 * K40;
    C00 -= S00 * K00;
    C01 -= S01 * K00;
    C11 -= S01 * K10;
    C22 -= S12 * K21;
    C23 -= S13 * K21;
    C33 -= S13 * K31;
    C44 -= S04 * K40;
    C40 -= S04 * K00;
    C41 -= S04 * K10;
    states_.emplace_back(State(state, {x0, x1, x2, x3, x4, C00, C11, C22, C33, C01, C23, C44, C40, C41}));
    state = &states_.back();
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

}  // namespace trklet
