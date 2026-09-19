#include "L1Trigger/TrackFindingTracklet/interface/State.h"

#include <cmath>
#include <vector>
#include <deque>
#include <algorithm>
#include <iterator>

namespace trklet {

  //
  State::Stub::Stub(KalmanFilterFormats* kff, const tt::FrameStub& frame) : stubDR_(frame, kff->dataFormats()) {
    v0_ = kff->format(VariableKF::v0).digi(std::pow(stubDR_.dPhi(), 2) / 3.);
    v1_ = kff->format(VariableKF::v1).digi(std::pow(stubDR_.dZ(), 2) / 3.);
  }

  // proto state constructor
  State::State(KalmanFilterFormats* kff,
               TrackDR* trackDR,
               int trackId,
               const std::vector<Stub*>& seed,
               const std::vector<Stub*>& proj)
      : kff_(kff),
        setup_(kff->setup()),
        trackDR_(trackDR),
        seed_(seed),
        proj_(proj),
        trackId_(trackId),
        parent_(nullptr),
        hitPattern_(0, setup_->kfNumProj()),
        trackPattern_(0, setup_->kfNumProj()),
        x0_(0.),
        x1_(0.),
        x2_(0.),
        x3_(0.),
        x4_(0.),
        C00_(9.e3),
        C01_(0.),
        C11_(9.e3),
        C22_(9.e3),
        C23_(0.),
        C33_(9.e3),
        C44_(9.e3),
        C40_(0.),
        C41_(0.) {
    for (int layer = 0; layer < static_cast<int>(proj_.size()); layer++)
      trackPattern_.set(layer);
  }

  // updated state constructor
  State::State(State* state, const std::vector<double>& doubles) : State(state) {
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
    const int layerThis = hitPattern_.count() == 0 ? -1 : hitPattern_.pmEncode();
    const int layerNext = trackPattern_.plEncode(layerThis + 1, setup_->kfNumProj());
    if (layerNext < setup_->kfNumProj())
      hitPattern_.set(layerNext);
  }

  // combinatoric and seed building state constructor
  State::State(State* state, State* parent, int layer) : State(state) {
    parent_ = parent;
    hitPattern_ = parent->hitPattern();
    hitPattern_.set(layer);
  }

  //
  State* State::comb(std::deque<State>& states, int layer) {
    // handle skipping
    if (!hitPattern_.test(layer))
      return nullptr;
    // handle skip
    const int nextLayer = trackPattern_.plEncode(layer + 1, setup_->kfNumProj());
    if (nextLayer == setup_->kfNumProj())
      return nullptr;
    // not enough layer left
    if (hitPattern_.count() - 1 + trackPattern_.count(nextLayer, setup_->kfNumProj()) < setup_->kfMinProj())
      return nullptr;
    states.emplace_back(this, parent_, nextLayer);
    return &states.back();
  }

  //
  State* State::final(std::deque<State>& states, int layer) {
    if (hitPattern_.test(layer) && hitPattern_.count(0, layer + 1) >= setup_->kfMinProj()) {
      states.emplace_back(this, parent_, layer);
      return &states.back();
    }
    return nullptr;
  }

  // copy constructor
  State::State(State* state)
      : kff_(state->kff_),
        setup_(state->setup_),
        trackDR_(state->trackDR_),
        seed_(state->seed_),
        proj_(state->proj_),
        trackId_(state->trackId_),
        parent_(state->parent_),
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
