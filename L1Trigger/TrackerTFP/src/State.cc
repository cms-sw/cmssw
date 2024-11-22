#include "L1Trigger/TrackerTFP/interface/State.h"

using namespace std;
using namespace tt;

namespace trackerTFP {

  //
  State::Stub::Stub(KalmanFilterFormats* formats, const FrameStub& frame) : stubCTB_(frame, formats->dataFormats()) {
    const Setup* setup = formats->setup();
    H12_ = formats->format(VariableKF::H12).digi(stubCTB_.r() + setup->chosenRofPhi() - setup->chosenRofZ());
    v0_ = formats->format(VariableKF::v0).digi(pow(2. * stubCTB_.dPhi(), 2));
    v1_ = formats->format(VariableKF::v1).digi(pow(2. * stubCTB_.dZ(), 2));
  }

  // proto state constructor
  State::State(KalmanFilterFormats* formats,
               TrackCTB* track,
               const vector<vector<Stub*>>& stubs,
               const TTBV& maybePattern,
               int trackId)
      : formats_(formats),
        setup_(formats->setup()),
        track_(track),
        stubs_(stubs),
        maybePattern_(maybePattern),
        trackId_(trackId),
        hitPattern_(0, setup_->numLayers()),
        trackPattern_(0, setup_->numLayers()) {
    for (const vector<Stub*>& stubs : stubs_) {
      if (!stubs.empty())
        trackPattern_.set(layer_);
      layer_++;
    }
    layer_ = trackPattern_.plEncode();
    stub_ = stubs_[layer_].front();
    hitPattern_.set(layer_);
  }

  // updated state constructor
  State::State(State* state, const vector<double>& doubles) : State(state) {
    parent_ = state;
    // updated track parameter and uncertainties
    x0_ = doubles[0];
    x1_ = doubles[1];
    x2_ = doubles[2];
    x3_ = doubles[3];
    chi20_ = doubles[4];
    chi21_ = doubles[5];
    C00_ = doubles[6];
    C11_ = doubles[7];
    C22_ = doubles[8];
    C33_ = doubles[9];
    C01_ = doubles[10];
    C23_ = doubles[11];
    // pick next stub (first stub in next layer with stub)
    stub_ = nullptr;
    if (hitPattern_.count() >= setup_->kfMinLayers() || hitPattern_.count() == setup_->kfMaxLayers()) {
      layer_ = 0;
      return;
    }
    layer_ = trackPattern_.plEncode(layer_ + 1, setup_->numLayers());
    if (layer_ == setup_->numLayers())
      return;
    stub_ = stubs_[layer_].front();
    hitPattern_.set(layer_);
  }

  // combinatoric and seed building state constructor
  State::State(State* state, State* parent, Stub* stub, int layer) : State(state) {
    parent_ = parent;
    stub_ = stub;
    layer_ = layer;
    hitPattern_ = parent ? parent->hitPattern() : TTBV(0, setup_->numLayers());
    hitPattern_.set(layer_);
  }

  //
  State* State::update(deque<State>& states, int layer) {
    if (!hitPattern_.test(layer) || hitPattern_.count() > setup_->kfNumSeedStubs())
      return this;
    layer_ = trackPattern_.plEncode(layer_ + 1, setup_->numLayers());
    states.emplace_back(this, this, stubs_[layer_].front(), layer_);
    return &states.back();
  }

  //
  State* State::combSeed(deque<State>& states, int layer) {
    // handle trivial state
    if (!hitPattern_.test(layer) || hitPattern_.count() > setup_->kfNumSeedStubs())
      return nullptr;
    // pick next stub on layer
    const vector<Stub*>& stubs = stubs_[layer];
    const int pos = distance(stubs.begin(), find(stubs.begin(), stubs.end(), stub_)) + 1;
    if (pos < (int)stubs.size()) {
      states.emplace_back(this, parent_, stubs[pos], layer);
      return &states.back();
    }
    // skip this layer
    const int nextLayer = trackPattern_.plEncode(layer + 1, setup_->numLayers());
    if (gapCheck(nextLayer)) {
      states.emplace_back(this, parent_, stubs_[nextLayer].front(), nextLayer);
      return &states.back();
    }
    return nullptr;
  }

  //
  State* State::comb(deque<State>& states, int layer) {
    // handle skipping and min reached
    if (!hitPattern_.test(layer)) {
      if (!stub_ && trackPattern_[layer] && hitPattern_.count() < setup_->kfMaxLayers()) {
        states.emplace_back(this, parent_, stubs_[layer].front(), layer);
        return &states.back();
      }
      return nullptr;
    }
    // handle part of seed
    if (hitPattern_.pmEncode() != layer)
      return nullptr;
    // handle multiple stubs on layer
    const vector<Stub*>& stubs = stubs_[layer];
    const int pos = distance(stubs.begin(), find(stubs.begin(), stubs.end(), stub_)) + 1;
    if (pos < (int)stubs.size()) {
      states.emplace_back(this, parent_, stubs[pos], layer);
      return &states.back();
    }
    // handle skip
    const int nextLayer = trackPattern_.plEncode(layer + 1, setup_->numLayers());
    if (gapCheck(nextLayer)) {
      states.emplace_back(this, parent_, stubs_[nextLayer].front(), nextLayer);
      return &states.back();
    }
    return nullptr;
  }

  //
  bool State::gapCheck(int layer) const {
    if (layer >= setup_->numLayers())
      return false;
    bool gap(false);
    int hits(0);
    int gaps(0);
    for (int k = 0; k < setup_->numLayers(); k++) {
      if (k == setup_->kfMaxSeedingLayer())
        if (hits < setup_->kfNumSeedStubs())
          return false;
      if (hitPattern_[k]) {
        gap = false;
        if (++hits >= setup_->kfMinLayers() && k >= layer)
          return true;
      } else if (!maybePattern_[k]) {
        if (gap || ++gaps > setup_->kfMaxGaps())
          return false;
        gap = true;
      }
    }
    return false;
  }

  // copy constructor
  State::State(State* state)
      : formats_(state->formats_),
        setup_(state->setup_),
        track_(state->track_),
        stubs_(state->stubs_),
        maybePattern_(state->maybePattern_),
        trackId_(state->trackId_),
        parent_(state->parent_),
        stub_(state->stub_),
        layer_(state->layer_),
        hitPattern_(state->hitPattern_),
        trackPattern_(state->trackPattern_),
        x0_(state->x0_),
        x1_(state->x1_),
        x2_(state->x2_),
        x3_(state->x3_),
        chi20_(state->chi20_),
        chi21_(state->chi21_),
        C00_(state->C00_),
        C01_(state->C01_),
        C11_(state->C11_),
        C22_(state->C22_),
        C23_(state->C23_),
        C33_(state->C33_) {}

}  // namespace trackerTFP
