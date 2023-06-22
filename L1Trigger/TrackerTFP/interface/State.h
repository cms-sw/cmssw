#ifndef L1Trigger_TrackerTFP_State_h
#define L1Trigger_TrackerTFP_State_h

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/DataFormats.h"

#include <vector>
#include <numeric>

namespace trackerTFP {

  // Class to represent a Kalman Filter State
  class State {
  public:
    // default constructor
    State(State* state);
    // proto state constructor
    State(const DataFormats* dataFormats, TrackKFin* track, int trackId);
    // combinatoric state constructor
    State(State* state, StubKFin* stub);
    // updated state constructor
    State(State* state, const std::vector<double>& doubles);
    ~State() {}

    // Determine quality of completed state
    void finish();
    // number of skipped layers
    int numSkippedLayers() const { return numSkippedLayers_; }
    // number of consitent layers
    int numConsistentLayers() const { return numConsistentLayers_; }
    // input track
    TrackKFin* track() const { return track_; }
    // parent state (nullpointer if no parent available)
    State* parent() const { return parent_; }
    // stub to add to state
    StubKFin* stub() const { return stub_; }
    // hitPattern of so far added stubs
    const TTBV& hitPattern() const { return hitPattern_; }
    // track id of input track
    int trackId() const { return trackId_; }
    // pattern of maybe layers for input track
    TTBV maybePattern() const { return track_->maybePattern(); }
    // stub id per layer of so far added stubs
    const std::vector<int>& layerMap() const { return layerMap_; }
    // layer id of the current stub to add
    int layer() const { return stub_->layer(); }
    // helix inv2R wrt input helix
    double x0() const { return x0_; }
    // helix phi at radius ChosenRofPhi wrt input helix
    double x1() const { return x1_; }
    // helix cot(Theta) wrt input helix
    double x2() const { return x2_; }
    // helix z at radius chosenRofZ wrt input helix
    double x3() const { return x3_; }
    // cov. matrix element
    double C00() const { return C00_; }
    // cov. matrix element
    double C01() const { return C01_; }
    // cov. matrix element
    double C11() const { return C11_; }
    // cov. matrix element
    double C22() const { return C22_; }
    // cov. matrix element
    double C23() const { return C23_; }
    // cov. matrix element
    double C33() const { return C33_; }
    // Derivative of predicted stub coords wrt helix params: stub radius minus chosenRofPhi
    double H00() const { return stub_->r(); }
    // Derivative of predicted stub coords wrt helix params: stub radius minus chosenRofZ
    double H12() const { return stub_->r() + dataFormats_->chosenRofPhi() - setup_->chosenRofZ(); }
    // stub phi residual wrt input helix
    double m0() const { return stub_->phi(); }
    // stub z residual wrt input helix
    double m1() const { return stub_->z(); }
    // stub projected phi uncertainty
    double dPhi() const { return stub_->dPhi(); }
    // stub projected z uncertainty
    double dZ() const { return stub_->dZ(); }
    // squared stub projected phi uncertainty instead of wheight (wrong but simpler)
    double v0() const { return pow(stub_->dPhi(), 2); }
    // squared stub projected z uncertainty instead of wheight (wrong but simpler)
    double v1() const { return pow(stub_->dZ(), 2); }
    // output frame
    tt::FrameTrack frame() const { return TrackKF(*track_, x1_, x0_, x3_, x2_).frame(); }
    // fill collection of stubs added so far to state
    void fill(std::vector<StubKF>& stubs) const;

  private:
    // provides data fomats
    const DataFormats* dataFormats_;
    // provides run-time constants
    const tt::Setup* setup_;
    // input track
    TrackKFin* track_;
    // track id
    int trackId_;
    // previous state, nullptr for first states
    State* parent_;
    // stub to add
    StubKFin* stub_;
    // shows which stub on each layer has been added so far
    std::vector<int> layerMap_;
    // shows which layer has been added so far
    TTBV hitPattern_;
    // helix inv2R wrt input helix
    double x0_;
    // helix phi at radius ChosenRofPhi wrt input helix
    double x1_;
    // helix cot(Theta) wrt input helix
    double x2_;
    // helix z at radius chosenRofZ wrt input helix
    double x3_;

    // cov. matrix

    double C00_;
    double C01_;
    double C11_;
    double C22_;
    double C23_;
    double C33_;

    // number of skipped layers
    int numSkippedLayers_;
    // number of consistent layers
    int numConsistentLayers_;
  };

}  // namespace trackerTFP

#endif