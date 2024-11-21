#ifndef L1Trigger_TrackFindingTracklet_State_h
#define L1Trigger_TrackFindingTracklet_State_h

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackFindingTracklet/interface/DataFormats.h"
#include "L1Trigger/TrackFindingTracklet/interface/KalmanFilterFormats.h"

#include <vector>
#include <numeric>

namespace trklet {

  // Class to represent a Kalman Filter helix State
  class State {
  public:
    //
    struct Stub {
      Stub(KalmanFilterFormats* kff, const tt::FrameStub& frame);
      StubDR stubDR_;
      double H12_;
      double H04_;
      double v0_;
      double v1_;
    };
    // copy constructor
    State(State* state);
    // proto state constructor
    State(KalmanFilterFormats* kff, TrackDR* track, const std::vector<Stub*>& stubs, int trackId);
    // updated state constructor
    State(State* state, const std::vector<double>& doubles);
    // combinatoric and seed building state constructor
    State(State* state, State* parent, int layer);
    ~State() {}
    //
    State* comb(std::deque<State>& states, int layer);
    //
    State* combSeed(std::deque<State>& states, int layer);
    //
    State* update(std::deque<State>& states, int layer);
    // input track
    TrackDR* track() const { return track_; }
    // parent state (nullpointer if no parent available)
    State* parent() const { return parent_; }
    // stub to add to state
    Stub* stub() const { return stub_; }
    // hitPattern of so far added stubs
    const TTBV& hitPattern() const { return hitPattern_; }
    // shows which layer the found track has stubs on
    const TTBV& trackPattern() const { return trackPattern_; }
    // track id of input track
    int trackId() const { return trackId_; }
    // helix inv2R wrt input helix
    double x0() const { return x0_; }
    // helix phi at radius ChosenRofPhi wrt input helix
    double x1() const { return x1_; }
    // helix cot(Theta) wrt input helix
    double x2() const { return x2_; }
    // helix z at radius chosenRofZ wrt input helix
    double x3() const { return x3_; }
    //
    double x4() const { return x4_; }
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
    double C44() const { return C44_; }
    double C40() const { return C40_; }
    double C41() const { return C41_; }
    // Derivative of predicted stub coords wrt helix params: stub radius minus chosenRofPhi
    double H00() const { return stub_->stubDR_.r(); }
    // Derivative of predicted stub coords wrt helix params: stub radius minus chosenRofZ
    double H12() const { return stub_->H12_; }
    //
    double H04() const { return stub_->H04_; }
    // stub phi residual wrt input helix
    double m0() const { return stub_->stubDR_.phi(); }
    // stub z residual wrt input helix
    double m1() const { return stub_->stubDR_.z(); }
    // stub projected phi uncertainty
    double d0() const { return stub_->stubDR_.dPhi(); }
    // stub projected z uncertainty
    double d1() const { return stub_->stubDR_.dZ(); }
    // squared stub projected phi uncertainty instead of wheight (wrong but simpler)
    double v0() const { return stub_->v0_; }
    // squared stub projected z uncertainty instead of wheight (wrong but simpler)
    double v1() const { return stub_->v1_; }
    // layer of current to add stub
    int layer() const { return std::distance(stubs_.begin(), std::find(stubs_.begin(), stubs_.end(), stub_)); }
    //
    std::vector<Stub*> stubs() const { return stubs_; }

  private:
    // provides data fomats
    KalmanFilterFormats* kff_;
    // provides run-time constants
    const tt::Setup* setup_;
    // input track
    TrackDR* track_;
    // input track stubs
    std::vector<Stub*> stubs_;
    // track id
    int trackId_;
    // previous state, nullptr for first states
    State* parent_;
    // stub to add
    Stub* stub_;
    // shows which layer has been added so far
    TTBV hitPattern_;
    // shows which layer the found track has stubs on
    TTBV trackPattern_;
    // helix inv2R wrt input helix
    double x0_;
    // helix phi at radius ChosenRofPhi wrt input helix
    double x1_;
    // helix cot(Theta) wrt input helix
    double x2_;
    // helix z at radius chosenRofZ wrt input helix
    double x3_;
    // impact parameter in 1/cm
    double x4_;
    // cov. matrix
    double C00_;
    double C01_;
    double C11_;
    double C22_;
    double C23_;
    double C33_;
    double C44_;
    double C40_;
    double C41_;
  };

}  // namespace trklet

#endif