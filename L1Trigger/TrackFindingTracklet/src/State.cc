#include "L1Trigger/TrackFindingTracklet/interface/State.h"

#include <cmath>
#include <vector>
#include <deque>
#include <algorithm>
#include <iterator>

using namespace std;
using namespace tt;

namespace trklet {

  //
  State::Stub::Stub(KalmanFilterFormats* kff, const FrameStub& frame) : stubDR_(frame, kff->dataFormats()) {
    const Setup* setup = kff->setup();
    H12_ = kff->format(VariableKF::H12).digi(stubDR_.r() + setup->chosenRofPhi() - setup->chosenRofZ());
    H04_ = stubDR_.r() + setup->chosenRofPhi();
    v0_ = kff->format(VariableKF::v0).digi(pow(2. * stubDR_.dPhi(), 2));
    v1_ = kff->format(VariableKF::v1).digi(pow(2. * stubDR_.dZ(), 2));
  }

  // proto state constructor
  State::State(KalmanFilterFormats* kff, TrackDR* track, const vector<Stub*>& stubs, int trackId)
      : kff_(kff),
        setup_(kff->setup()),
        track_(track),
        stubs_(stubs),
        trackId_(trackId),
        parent_(nullptr),
        stub_(nullptr),
        hitPattern_(0, setup_->numLayers()),
        trackPattern_(0, setup_->numLayers()),
        x0_(0.),
        x1_(0.),
        x2_(0.),
        x3_(0.),
        x4_(0.),
        C00_(9.e9),
        C01_(0.),
        C11_(9.e9),
        C22_(9.e9),
        C23_(0.),
        C33_(9.e9),
        C44_(pow(setup_->maxD0(), 2)),
        C40_(0.),
        C41_(0.) {
    int layer(0);
    for (Stub* stub : stubs_)
      trackPattern_[layer++] = (bool)stub;
    layer = trackPattern_.plEncode();
    stub_ = stubs_[layer];
    hitPattern_.set(layer);
  }

  // updated state constructor
  State::State(State* state, const vector<double>& doubles) : State(state) {
    parent_ = state;
    // updated track parameter and uncertainties
    x0_ = doubles[0];
    x1_ = doubles[1];
    x2_ = doubles[2];
    x3_ = doubles[3];
    x4_ = doubles[4];
    C00_ = doubles[5];
    C11_ = doubles[6];
    C22_ = doubles[7];
    C33_ = doubles[8];
    C01_ = doubles[9];
    C23_ = doubles[10];
    C44_ = doubles[11];
    C40_ = doubles[12];
    C41_ = doubles[13];
    // pick next stub
    const int layer = this->layer();
    stub_ = nullptr;
    if (hitPattern_.count() >= setup_->kfMinLayers() || hitPattern_.count() == setup_->kfMaxLayers())
      return;
    const int nextLayer = trackPattern_.plEncode(layer + 1, setup_->numLayers());
    if (nextLayer == setup_->numLayers())
      return;
    stub_ = stubs_[nextLayer];
    hitPattern_.set(nextLayer);
  }

  // combinatoric and seed building state constructor
  State::State(State* state, State* parent, int layer) : State(state) {
    parent_ = parent;
    hitPattern_ = parent ? parent->hitPattern() : TTBV(0, setup_->numLayers());
    stub_ = stubs_[layer];
    hitPattern_.set(layer);
  }

  //
  State* State::update(deque<State>& states, int layer) {
    if (!hitPattern_.test(layer) || hitPattern_.count() > setup_->kfNumSeedStubs())
      return this;
    const int nextLayer = trackPattern_.plEncode(layer + 1, setup_->numLayers());
    states.emplace_back(this, this, nextLayer);
    return &states.back();
  }

  //
  State* State::combSeed(deque<State>& states, int layer) {
    // handle trivial state
    if (!hitPattern_.test(layer) || hitPattern_.count() > setup_->kfNumSeedStubs())
      return nullptr;
    // skip layers
    const int nextLayer = trackPattern_.plEncode(layer + 1, setup_->numLayers());
    const int maxSeedStubs = hitPattern_.count(0, layer) + trackPattern_.count(nextLayer, setup_->kfMaxSeedingLayer());
    if (maxSeedStubs < setup_->kfNumSeedStubs())
      return nullptr;
    const int maxStubs = maxSeedStubs + trackPattern_.count(setup_->kfMaxSeedingLayer(), setup_->numLayers());
    if (maxStubs < setup_->kfMinLayers())
      return nullptr;
    states.emplace_back(this, parent_, nextLayer);
    return &states.back();
  }

  //
  State* State::comb(deque<State>& states, int layer) {
    // handle skipping and min reached
    if (!hitPattern_.test(layer)) {
      if (!stub_ && trackPattern_[layer] && hitPattern_.count() < setup_->kfMaxLayers()) {
        states.emplace_back(this, parent_, layer);
        return &states.back();
      }
      return nullptr;
    }
    // handle part of seed
    if (hitPattern_.pmEncode() != layer)
      return nullptr;
    // handle skip
    const int nextLayer = trackPattern_.plEncode(layer + 1, setup_->numLayers());
    if (nextLayer == setup_->numLayers())
      return nullptr;
    // not enough layer left
    if (hitPattern_.count() - 1 + trackPattern_.count(nextLayer, setup_->numLayers()) < setup_->kfMinLayers())
      return nullptr;
    states.emplace_back(this, parent_, nextLayer);
    return &states.back();
  }

  // copy constructor
  State::State(State* state)
      : kff_(state->kff_),
        setup_(state->setup_),
        track_(state->track_),
        stubs_(state->stubs_),
        trackId_(state->trackId_),
        parent_(state->parent_),
        stub_(state->stub_),
        hitPattern_(state->hitPattern_),
        trackPattern_(state->trackPattern_),
        x0_(state->x0_),
        x1_(state->x1_),
        x2_(state->x2_),
        x3_(state->x3_),
        x4_(state->x4_),
        C00_(state->C00_),
        C01_(state->C01_),
        C11_(state->C11_),
        C22_(state->C22_),
        C23_(state->C23_),
        C33_(state->C33_),
        C44_(state->C44_),
        C40_(state->C40_),
        C41_(state->C41_) {}

}  // namespace trklet
