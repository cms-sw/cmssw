#ifndef L1Trigger_TrackerTFP_State_h
#define L1Trigger_TrackerTFP_State_h

#include "L1Trigger/TrackTrigger/interface/Setup.h"
#include "L1Trigger/TrackerTFP/interface/KalmanFilterFormats.h"

#include <vector>
#include <numeric>

namespace trackerTFP {

  // Class to represent a Kalman Filter helix State
  class State {
  public:
    //
    struct Stub {
      Stub(KalmanFilterFormats* formats, const tt::FrameStub& frame);
      StubCTB stubCTB_;
      double H12_;
      double v0_;
      double v1_;
    };
    // copy constructor
    State(State* state);
    // proto state constructor
    State(KalmanFilterFormats* formats,
          TrackCTB* track,
          const std::vector<std::vector<Stub*>>& stubs,
          const TTBV& maybe,
          int trackId);
    // updated state constructor
    State(State* state, const std::vector<double>& doubles);
    // combinatoric and seed building state constructor
    State(State* state, State* parent, Stub* stub, int layer);
    ~State() {}
    //
    State* comb(std::deque<State>& states, int layer);
    //
    State* combSeed(std::deque<State>& states, int layer);
    //
    State* update(std::deque<State>& states, int layer);
    // input track
    TrackCTB* track() const { return track_; }
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
    // pattern of maybe layers for input track
    const TTBV& maybePattern() const { return maybePattern_; }
    // layer id of the current stub to add
    int layer() const { return layer_; }
    // helix inv2R wrt input helix
    double x0() const { return x0_; }
    // helix phi at radius ChosenRofPhi wrt input helix
    double x1() const { return x1_; }
    // helix cot(Theta) wrt input helix
    double x2() const { return x2_; }
    // helix z at radius chosenRofZ wrt input helix
    double x3() const { return x3_; }
    // chi2 for the r-phi plane straight line fit
    double chi20() const { return chi20_; }
    // chi2 for the r-z plane straight line fit
    double chi21() const { return chi21_; }
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
    double H00() const { return stub_->stubCTB_.r(); }
    // Derivative of predicted stub coords wrt helix params: stub radius minus chosenRofZ
    double H12() const { return stub_->H12_; }
    // stub phi residual wrt input helix
    double m0() const { return stub_->stubCTB_.phi(); }
    // stub z residual wrt input helix
    double m1() const { return stub_->stubCTB_.z(); }
    // stub projected phi uncertainty
    double dPhi() const { return stub_->stubCTB_.dPhi(); }
    // stub projected z uncertainty
    double dZ() const { return stub_->stubCTB_.dZ(); }
    // squared stub projected phi uncertainty instead of wheight (wrong but simpler)
    double v0() const { return stub_->v0_; }
    // squared stub projected z uncertainty instead of wheight (wrong but simpler)
    double v1() const { return stub_->v1_; }
    //const std::vector<std::vector<StubCTB*>>& stubs() const { return stubs_; }

  private:
    //
    bool gapCheck(int layer) const;
    // provides data fomats
    KalmanFilterFormats* formats_;
    // provides run-time constants
    const tt::Setup* setup_;
    // input track
    TrackCTB* track_;
    // input track stubs
    std::vector<std::vector<Stub*>> stubs_;
    // pattern of maybe layers for input track
    TTBV maybePattern_;
    // track id
    int trackId_;
    // previous state, nullptr for first states
    State* parent_ = nullptr;
    // stub to add
    Stub* stub_ = nullptr;
    // layer id of the current stub to add
    int layer_ = 0;
    // shows which layer has been added so far
    TTBV hitPattern_;
    // shows which layer the found track has stubs on
    TTBV trackPattern_;
    // helix inv2R wrt input helix
    double x0_ = 0.;
    // helix phi at radius ChosenRofPhi wrt input helix
    double x1_ = 0.;
    // helix cot(Theta) wrt input helix
    double x2_ = 0.;
    // helix z at radius chosenRofZ wrt input helix
    double x3_ = 0.;
    // chi2 for the r-phi plane straight line fit
    double chi20_ = 0.;
    // chi2 for the r-z plane straight line fit
    double chi21_ = 0.;
    // cov. matrix
    double C00_ = 9.e9;
    double C01_ = 0.;
    double C11_ = 9.e9;
    double C22_ = 9.e9;
    double C23_ = 0.;
    double C33_ = 9.e9;
  };

}  // namespace trackerTFP

#endif
