#include "L1Trigger/TrackerTFP/interface/State.h"

namespace trackerTFP {

  //
  State::Stub::Stub(KalmanFilterFormats* formats, const tt::FrameStub& frame)
      : stubCTB_(frame, formats->dataFormats()) {
    const tt::Setup* setup = formats->setup();
    H12_ = formats->format(VariableKF::H12).digi(stubCTB_.r() + setup->chosenRofPhi() - setup->chosenRofZ());
    v0_ = formats->format(VariableKF::v0).digi(std::pow(2. * stubCTB_.dPhi(), 2));
    v1_ = formats->format(VariableKF::v1).digi(std::pow(2. * stubCTB_.dZ(), 2));
  }

  // proto state constructor
  State::State(KalmanFilterFormats* formats,
               TrackCTB* track,
               const std::vector<std::vector<Stub*>>& stubs,
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
    for (const std::vector<Stub*>& stubs : stubs_) {
      if (!stubs.empty())
        trackPattern_.set(layer_);
      layer_++;
    }
    layer_ = trackPattern_.plEncode();
    stub_ = stubs_[layer_].front();
    hitPattern_.set(layer_);
    chi20_ = formats_->format(VariableKF::chi20).digi(0);
    chi21_ = formats_->format(VariableKF::chi21).digi(0);
  }

  // updated state constructor
  State::State(State* state,
               double x0,
               double x1,
               double x2,
               double x3,
               double C00,
               double C01,
               double C11,
               double C22,
               double C23,
               double C33,
               double chi20,
               double chi21)
      : State(state) {
    parent_ = state;
    // updated track parameter and uncertainties
    x0_ = x0;
    x1_ = x1;
    x2_ = x2;
    x3_ = x3;
    chi20_ = chi20;
    chi21_ = chi21;
    C00_ = C00;
    C11_ = C11;
    C22_ = C22;
    C33_ = C33;
    C01_ = C01;
    C23_ = C23;
    // pick next stub (first stub in next layer with stub)
    stub_ = nullptr;
    layer_ = trackPattern_.plEncode(layer_ + 1, setup_->numLayers());
    if (hitPattern_.count() >= setup_->kfMinLayers() || hitPattern_.count() == setup_->kfMaxLayers()) {
      layer_ = 0;
      return;
    }
    if (layer_ == setup_->numLayers())
      return;
    stub_ = stubs_[layer_].front();
    hitPattern_.set(layer_);
  }

  // seed state constructor
  State::State(State* state,
               double x0,
               double x1,
               double x2,
               double x3,
               double C00,
               double C01,
               double C11,
               double C22,
               double C23,
               double C33)
      : State(state) {
    // seed track parameter and uncertainties
    x0_ = x0;
    x1_ = x1;
    x2_ = x2;
    x3_ = x3;
    C00_ = C00;
    C11_ = C11;
    C22_ = C22;
    C33_ = C33;
    C01_ = C01;
    C23_ = C23;
  }

  // combinatoric state constructor
  State::State(State* state, State* parent, Stub* stub, int layer) : State(state) {
    parent_ = parent;
    stub_ = stub;
    layer_ = layer;
    hitPattern_ = parent ? parent->hitPattern() : TTBV(0, setup_->numLayers());
    hitPattern_.set(layer_);
  }

  //
  State* State::update(std::deque<State>& states, int layer) {
    if (!hitPattern_.test(layer) || hitPattern_.count() > setup_->kfNumSeedStubs())
      return this;
    layer_ = trackPattern_.plEncode(layer_ + 1, setup_->numLayers());
    states.emplace_back(this, this, stubs_[layer_].front(), layer_);
    return &states.back();
  }

  //
  State* State::combSeed(std::deque<State>& states, int layer) {
    // handle trivial state
    if (!hitPattern_.test(layer) || hitPattern_.count() > setup_->kfNumSeedStubs())
      return nullptr;
    // pick next stub on layer
    const std::vector<Stub*>& stubs = stubs_[layer];
    const int pos = std::distance(stubs.begin(), std::find(stubs.begin(), stubs.end(), stub_)) + 1;
    if (pos < static_cast<int>(stubs.size())) {
      states.emplace_back(this, parent_, stubs[pos], layer);
      return &states.back();
    }
    // skip this layer
    if (gapCheck(layer)) {
      const int nextLayer = trackPattern_.plEncode(layer + 1, setup_->numLayers());
      states.emplace_back(this, parent_, stubs_[nextLayer].front(), nextLayer);
      return &states.back();
    }
    return nullptr;
  }

  //
  State* State::comb(std::deque<State>& states, int layer) {
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
    const std::vector<Stub*>& stubs = stubs_[layer];
    const int pos = std::distance(stubs.begin(), find(stubs.begin(), stubs.end(), stub_)) + 1;
    if (pos < static_cast<int>(stubs.size())) {
      states.emplace_back(this, parent_, stubs[pos], layer);
      return &states.back();
    }
    // handle skip
    if (gapCheck(layer)) {
      const int nextLayer = trackPattern_.plEncode(layer + 1, setup_->numLayers());
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
      const TTBV& pattern = k < layer ? hitPattern_ : trackPattern_;
      if (k != layer && pattern[k]) {
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
