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
                             KalmanFilterFormats* kalmanFilterFormats,
                             int region)
      : enableTruncation_(iConfig.getParameter<bool>("EnableTruncation")),
        setup_(setup),
        dataFormats_(dataFormats),
        kalmanFilterFormats_(kalmanFilterFormats),
        region_(region),
        input_(dataFormats_->numChannel(Process::kf)),
        layer_(0),
        x0_(&kalmanFilterFormats_->format(VariableKF::x0)),
        x1_(&kalmanFilterFormats_->format(VariableKF::x1)),
        x2_(&kalmanFilterFormats_->format(VariableKF::x2)),
        x3_(&kalmanFilterFormats_->format(VariableKF::x3)),
        H00_(&kalmanFilterFormats_->format(VariableKF::H00)),
        H12_(&kalmanFilterFormats_->format(VariableKF::H12)),
        m0_(&kalmanFilterFormats_->format(VariableKF::m0)),
        m1_(&kalmanFilterFormats_->format(VariableKF::m1)),
        v0_(&kalmanFilterFormats_->format(VariableKF::v0)),
        v1_(&kalmanFilterFormats_->format(VariableKF::v1)),
        r0_(&kalmanFilterFormats_->format(VariableKF::r0)),
        r1_(&kalmanFilterFormats_->format(VariableKF::r1)),
        S00_(&kalmanFilterFormats_->format(VariableKF::S00)),
        S01_(&kalmanFilterFormats_->format(VariableKF::S01)),
        S12_(&kalmanFilterFormats_->format(VariableKF::S12)),
        S13_(&kalmanFilterFormats_->format(VariableKF::S13)),
        K00_(&kalmanFilterFormats_->format(VariableKF::K00)),
        K10_(&kalmanFilterFormats_->format(VariableKF::K10)),
        K21_(&kalmanFilterFormats_->format(VariableKF::K21)),
        K31_(&kalmanFilterFormats_->format(VariableKF::K31)),
        R00_(&kalmanFilterFormats_->format(VariableKF::R00)),
        R11_(&kalmanFilterFormats_->format(VariableKF::R11)),
        R00Rough_(&kalmanFilterFormats_->format(VariableKF::R00Rough)),
        R11Rough_(&kalmanFilterFormats_->format(VariableKF::R11Rough)),
        invR00Approx_(&kalmanFilterFormats_->format(VariableKF::invR00Approx)),
        invR11Approx_(&kalmanFilterFormats_->format(VariableKF::invR11Approx)),
        invR00Cor_(&kalmanFilterFormats_->format(VariableKF::invR00Cor)),
        invR11Cor_(&kalmanFilterFormats_->format(VariableKF::invR11Cor)),
        invR00_(&kalmanFilterFormats_->format(VariableKF::invR00)),
        invR11_(&kalmanFilterFormats_->format(VariableKF::invR11)),
        C00_(&kalmanFilterFormats_->format(VariableKF::C00)),
        C01_(&kalmanFilterFormats_->format(VariableKF::C01)),
        C11_(&kalmanFilterFormats_->format(VariableKF::C11)),
        C22_(&kalmanFilterFormats_->format(VariableKF::C22)),
        C23_(&kalmanFilterFormats_->format(VariableKF::C23)),
        C33_(&kalmanFilterFormats_->format(VariableKF::C33)) {
    C00_->updateRangeActual(pow(dataFormats_->base(Variable::inv2R, Process::kfin), 2));
    C11_->updateRangeActual(pow(dataFormats_->base(Variable::phiT, Process::kfin), 2));
    C22_->updateRangeActual(pow(dataFormats_->base(Variable::cot, Process::kfin), 2));
    C33_->updateRangeActual(pow(dataFormats_->base(Variable::zT, Process::kfin), 2));
  }

  // read in and organize input product (fill vector input_)
  void KalmanFilter::consume(const StreamsTrack& streamsTrack, const StreamsStub& streamsStub) {
    auto valid = [](const auto& frame) { return frame.first.isNonnull(); };
    auto acc = [](int& sum, const auto& frame) { return sum += (frame.first.isNonnull() ? 1 : 0); };
    int nTracks(0);
    int nStubs(0);
    const int offset = region_ * dataFormats_->numChannel(Process::kf);
    for (int channel = 0; channel < dataFormats_->numChannel(Process::kf); channel++) {
      const int channelTrack = offset + channel;
      const StreamTrack& streamTracks = streamsTrack[channelTrack];
      nTracks += accumulate(streamTracks.begin(), streamTracks.end(), 0, acc);
      for (int layer = 0; layer < setup_->numLayers(); layer++) {
        const int channelStub = channelTrack * setup_->numLayers() + layer;
        const StreamStub& streamStubs = streamsStub[channelStub];
        nStubs += accumulate(streamStubs.begin(), streamStubs.end(), 0, acc);
      }
    }
    tracks_.reserve(nTracks);
    stubs_.reserve(nStubs);
    // N.B. One input stream for track & one for its stubs in each layer. If a track has N stubs in one layer, and fewer in all other layers, then next valid track will be N frames later
    for (int channel = 0; channel < dataFormats_->numChannel(Process::kf); channel++) {
      const int channelTrack = offset + channel;
      const StreamTrack& streamTracks = streamsTrack[channelTrack];
      vector<TrackKFin*>& tracks = input_[channel];
      tracks.reserve(streamTracks.size());
      for (int frame = 0; frame < (int)streamTracks.size(); frame++) {
        const FrameTrack& frameTrack = streamTracks[frame];
        // Select frames with valid track
        if (frameTrack.first.isNull()) {
          if (dataFormats_->hybrid())
            tracks.push_back(nullptr);
          continue;
        }
        auto endOfTrk = find_if(next(streamTracks.begin(), frame + 1), streamTracks.end(), valid);
        if (dataFormats_->hybrid())
          endOfTrk = next(streamTracks.begin(), frame + 1);
        // No. of frames before next track indicates gives max. no. of stubs this track had in any layer
        const int maxStubsPerLayer = distance(next(streamTracks.begin(), frame), endOfTrk);
        tracks.insert(tracks.end(), maxStubsPerLayer - 1, nullptr);
        deque<StubKFin*> stubs;
        for (int layer = 0; layer < setup_->numLayers(); layer++) {
          const int channelStub = channelTrack * setup_->numLayers() + layer;
          const StreamStub& streamStubs = streamsStub[channelStub];
          // Get stubs on this track
          for (int i = frame; i < frame + maxStubsPerLayer; i++) {
            const FrameStub& frameStub = streamStubs[i];
            if (frameStub.first.isNull())
              break;
            // Store input stubs, so remainder of KF algo can work with pointers to them (saves CPU)
            stubs_.emplace_back(frameStub, dataFormats_, layer);
            stubs.push_back(&stubs_.back());
          }
        }
        // Store input tracks, so remainder of KF algo can work with pointers to them (saves CPU)
        tracks_.emplace_back(frameTrack, dataFormats_, vector<StubKFin*>(stubs.begin(), stubs.end()));
        tracks.push_back(&tracks_.back());
      }
    }
  }

  // fill output products
  void KalmanFilter::produce(StreamsStub& acceptedStubs,
                             StreamsTrack& acceptedTracks,
                             StreamsStub& lostStubs,
                             StreamsTrack& lostTracks,
                             int& numAcceptedStates,
                             int& numLostStates) {
    auto put = [this](
                   const deque<State*>& states, StreamsStub& streamsStubs, StreamsTrack& streamsTracks, int channel) {
      const int streamId = region_ * dataFormats_->numChannel(Process::kf) + channel;
      const int offset = streamId * setup_->numLayers();
      StreamTrack& tracks = streamsTracks[streamId];
      tracks.reserve(states.size());
      for (int layer = 0; layer < setup_->numLayers(); layer++)
        streamsStubs[offset + layer].reserve(states.size());
      for (State* state : states) {
        tracks.emplace_back(state->frame());
        vector<StubKF> stubs;
        state->fill(stubs);
        for (const StubKF& stub : stubs)
          streamsStubs[offset + stub.layer()].emplace_back(stub.frame());
        // adding a gap to all layer without a stub
        for (int layer : state->hitPattern().ids(false))
          streamsStubs[offset + layer].emplace_back(FrameStub());
      }
    };
    auto count = [this](int& sum, const State* state) {
      return sum += state && state->hitPattern().count() >= setup_->kfMinLayers() ? 1 : 0;
    };
    for (int channel = 0; channel < dataFormats_->numChannel(Process::kf); channel++) {
      deque<State*> stream;
      deque<State*> lost;
      // proto state creation
      int trackId(0);
      for (TrackKFin* track : input_[channel]) {
        State* state = nullptr;
        if (track) {
          // Store states, so remainder of KF algo can work with pointers to them (saves CPU)
          states_.emplace_back(dataFormats_, track, trackId++);
          state = &states_.back();
        }
        stream.push_back(state);
      }
      // Propagate state to each layer in turn, updating it with all viable stub combinations there, using KF maths
      for (layer_ = 0; layer_ < setup_->numLayers(); layer_++)
        addLayer(stream);
      // calculate number of states before truncating
      const int numUntruncatedStates = accumulate(stream.begin(), stream.end(), 0, count);
      // untruncated best state selection
      deque<State*> untruncatedStream = stream;
      accumulator(untruncatedStream);
      // apply truncation
      if (enableTruncation_ && (int)stream.size() > setup_->numFrames())
        stream.resize(setup_->numFrames());
      // calculate number of states after truncating
      const int numTruncatedStates = accumulate(stream.begin(), stream.end(), 0, count);
      // best state per candidate selection
      accumulator(stream);
      deque<State*> truncatedStream = stream;
      // storing of best states missed due to truncation
      sort(untruncatedStream.begin(), untruncatedStream.end());
      sort(truncatedStream.begin(), truncatedStream.end());
      set_difference(untruncatedStream.begin(),
                     untruncatedStream.end(),
                     truncatedStream.begin(),
                     truncatedStream.end(),
                     back_inserter(lost));
      // store found tracks
      put(stream, acceptedStubs, acceptedTracks, channel);
      // store lost tracks
      put(lost, lostStubs, lostTracks, channel);
      // store number of states which got taken into account
      numAcceptedStates += numTruncatedStates;
      // store number of states which got not taken into account due to truncation
      numLostStates += numUntruncatedStates - numTruncatedStates;
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
    vector<State*> delay(latency, nullptr);
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
        // Assign next combinatoric stub to state
        comb(state);
      delay.push_back(state);
      state = pop_front(delay);
      if (state)
        stack.push_back(state);
    }
    stream = streamOutput;
    // Update state with next stub using KF maths
    for (State*& state : stream)
      if (state && state->stub() && state->layer() == layer_)
        update(state);
  }

  // Assign next combinatoric (i.e. not first in layer) stub to state
  void KalmanFilter::comb(State*& state) {
    const TrackKFin* track = state->track();
    const StubKFin* stub = state->stub();
    const vector<StubKFin*>& stubs = track->layerStubs(layer_);
    const TTBV& hitPattern = state->hitPattern();
    StubKFin* stubNext = nullptr;
    bool valid = state->stub() && state->layer() == layer_;
    if (valid) {
      // Get next unused stub on this layer
      const int pos = distance(stubs.begin(), find(stubs.begin(), stubs.end(), stub)) + 1;
      if (pos != (int)stubs.size())
        stubNext = stubs[pos];
      // picks next stub on different layer, nullifies state if skipping layer is not valid
      else {
        // having already maximum number of added layers
        if (hitPattern.count() == setup_->kfMaxLayers())
          valid = false;
        // Impossible for this state to ever get enough layers to form valid track
        if (hitPattern.count() + track->hitPattern().count(stub->layer() + 1, setup_->numLayers()) <
            setup_->kfMinLayers())
          valid = false;
        // not diffrent layers left
        if (layer_ == setup_->numLayers() - 1)
          valid = false;
        if (valid) {
          // pick next stub on next populated layer
          for (int nextLayer = layer_ + 1; nextLayer < setup_->numLayers(); nextLayer++) {
            if (track->hitPattern(nextLayer)) {
              stubNext = track->layerStub(nextLayer);
              break;
            }
          }
        }
      }
    }
    if (valid) {
      // create combinatoric state
      states_.emplace_back(state, stubNext);
      state = &states_.back();
    } else
      state = nullptr;
  }

  // best state selection
  void KalmanFilter::accumulator(deque<State*>& stream) {
    // accumulator delivers contigious stream of best state per track
    // remove gaps and not final states
    stream.erase(
        remove_if(stream.begin(),
                  stream.end(),
                  [this](State* state) { return !state || state->hitPattern().count() < setup_->kfMinLayers(); }),
        stream.end());
    // Determine quality of completed state
    for (State* state : stream)
      state->finish();
    // sort in number of skipped layers
    auto lessSkippedLayers = [](State* lhs, State* rhs) { return lhs->numSkippedLayers() < rhs->numSkippedLayers(); };
    stable_sort(stream.begin(), stream.end(), lessSkippedLayers);
    // sort in number of consistent stubs
    auto moreConsistentLayers = [](State* lhs, State* rhs) {
      return lhs->numConsistentLayers() > rhs->numConsistentLayers();
    };
    stable_sort(stream.begin(), stream.end(), moreConsistentLayers);
    // sort in track id
    stable_sort(stream.begin(), stream.end(), [](State* lhs, State* rhs) { return lhs->trackId() < rhs->trackId(); });
    // keep first state (best due to previous sorts) per track id
    stream.erase(
        unique(stream.begin(), stream.end(), [](State* lhs, State* rhs) { return lhs->track() == rhs->track(); }),
        stream.end());
  }

  // updates state
  void KalmanFilter::update(State*& state) {
    // All variable names & equations come from Fruhwirth KF paper http://dx.doi.org/10.1016/0168-9002%2887%2990887-4", where F taken as unit matrix. Stub uncertainties projected onto (phi,z), assuming no correlations between r-phi & r-z planes.
    // stub phi residual wrt input helix
    const double m0 = m0_->digi(state->m0());
    // stub z residual wrt input helix
    const double m1 = m1_->digi(state->m1());
    // stub projected phi uncertainty squared);
    const double v0 = v0_->digi(state->v0());
    // stub projected z uncertainty squared
    const double v1 = v1_->digi(state->v1());
    // helix inv2R wrt input helix
    double x0 = x0_->digi(state->x0());
    // helix phi at radius ChosenRofPhi wrt input helix
    double x1 = x1_->digi(state->x1());
    // helix cot(Theta) wrt input helix
    double x2 = x2_->digi(state->x2());
    // helix z at radius chosenRofZ wrt input helix
    double x3 = x3_->digi(state->x3());
    // Derivative of predicted stub coords wrt helix params: stub radius minus chosenRofPhi
    const double H00 = H00_->digi(state->H00());
    // Derivative of predicted stub coords wrt helix params: stub radius minus chosenRofZ
    const double H12 = H12_->digi(state->H12());
    // cov. matrix
    double C00 = C00_->digi(state->C00());
    double C01 = C01_->digi(state->C01());
    double C11 = C11_->digi(state->C11());
    double C22 = C22_->digi(state->C22());
    double C23 = C23_->digi(state->C23());
    double C33 = C33_->digi(state->C33());
    // stub phi residual wrt current state
    const double r0C = x1_->digi(m0 - x1);
    const double r0 = r0_->digi(r0C - x0 * H00);
    // stub z residual wrt current state
    const double r1C = x3_->digi(m1 - x3);
    const double r1 = r1_->digi(r1C - x2 * H12);
    // matrix S = H*C
    const double S00 = S00_->digi(C01 + H00 * C00);
    const double S01 = S01_->digi(C11 + H00 * C01);
    const double S12 = S12_->digi(C23 + H12 * C22);
    const double S13 = S13_->digi(C33 + H12 * C23);
    // Cov. matrix of predicted residuals R = V+HCHt = C+H*St
    const double R00C = S01_->digi(v0 + S01);
    const double R00 = R00_->digi(R00C + H00 * S00);
    const double R11C = S13_->digi(v1 + S13);
    const double R11 = R11_->digi(R11C + H12 * S12);
    // imrpoved dynamic cancelling
    const int msb0 = max(0, (int)ceil(log2(R00 / R00_->base())));
    const int msb1 = max(0, (int)ceil(log2(R11 / R11_->base())));
    const double R00Rough = R00Rough_->digi(R00 * pow(2., 16 - msb0));
    const double invR00Approx = invR00Approx_->digi(1. / R00Rough);
    const double invR00Cor = invR00Cor_->digi(2. - invR00Approx * R00Rough);
    const double invR00 = invR00_->digi(invR00Approx * invR00Cor * pow(2., 16 - msb0));
    const double R11Rough = R11Rough_->digi(R11 * pow(2., 16 - msb1));
    const double invR11Approx = invR11Approx_->digi(1. / R11Rough);
    const double invR11Cor = invR11Cor_->digi(2. - invR11Approx * R11Rough);
    const double invR11 = invR11_->digi(invR11Approx * invR11Cor * pow(2., 16 - msb1));
    // Kalman gain matrix K = S*R(inv)
    const double K00 = K00_->digi(S00 * invR00);
    const double K10 = K10_->digi(S01 * invR00);
    const double K21 = K21_->digi(S12 * invR11);
    const double K31 = K31_->digi(S13 * invR11);
    // Updated helix params & their cov. matrix
    x0 = x0_->digi(x0 + r0 * K00);
    x1 = x1_->digi(x1 + r0 * K10);
    x2 = x2_->digi(x2 + r1 * K21);
    x3 = x3_->digi(x3 + r1 * K31);
    C00 = C00_->digi(C00 - S00 * K00);
    C01 = C01_->digi(C01 - S01 * K00);
    C11 = C11_->digi(C11 - S01 * K10);
    C22 = C22_->digi(C22 - S12 * K21);
    C23 = C23_->digi(C23 - S13 * K21);
    C33 = C33_->digi(C33 - S13 * K31);
    // create updated state
    states_.emplace_back(State(state, (initializer_list<double>){x0, x1, x2, x3, C00, C11, C22, C33, C01, C23}));
    state = &states_.back();
    // update variable ranges to tune variable granularity
    m0_->updateRangeActual(m0);
    m1_->updateRangeActual(m1);
    v0_->updateRangeActual(v0);
    v1_->updateRangeActual(v1);
    H00_->updateRangeActual(H00);
    H12_->updateRangeActual(H12);
    r0_->updateRangeActual(r0);
    r1_->updateRangeActual(r1);
    S00_->updateRangeActual(S00);
    S01_->updateRangeActual(S01);
    S12_->updateRangeActual(S12);
    S13_->updateRangeActual(S13);
    R00_->updateRangeActual(R00);
    R11_->updateRangeActual(R11);
    R00Rough_->updateRangeActual(R00Rough);
    invR00Approx_->updateRangeActual(invR00Approx);
    invR00Cor_->updateRangeActual(invR00Cor);
    invR00_->updateRangeActual(invR00);
    R11Rough_->updateRangeActual(R11Rough);
    invR11Approx_->updateRangeActual(invR11Approx);
    invR11Cor_->updateRangeActual(invR11Cor);
    invR11_->updateRangeActual(invR11);
    K00_->updateRangeActual(K00);
    K10_->updateRangeActual(K10);
    K21_->updateRangeActual(K21);
    K31_->updateRangeActual(K31);
    x0_->updateRangeActual(x0);
    x1_->updateRangeActual(x1);
    x2_->updateRangeActual(x2);
    x3_->updateRangeActual(x3);
    C00_->updateRangeActual(C00);
    C01_->updateRangeActual(C01);
    C11_->updateRangeActual(C11);
    C22_->updateRangeActual(C22);
    C23_->updateRangeActual(C23);
    C33_->updateRangeActual(C33);
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

  // remove and return first element of vector, returns nullptr if empty
  template <class T>
  T* KalmanFilter::pop_front(vector<T*>& ts) const {
    T* t = nullptr;
    if (!ts.empty()) {
      t = ts.front();
      ts.erase(ts.begin());
    }
    return t;
  }

}  // namespace trackerTFP