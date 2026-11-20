#ifndef L1Trigger_TrackFindingTracklet_State_h
#define L1Trigger_TrackFindingTracklet_State_h

#include "L1Trigger/TrackFindingTracklet/interface/Setup.h"
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
      Stub(KalmanFilterFormats*, const tt::FrameStub&);
      // stub radius in cm
      double H() const { return stubDR_.r(); }
      // stub phi coordinate in rad
      double m0() const { return stubDR_.phi(); }
      // stub z coordinate in cm
      double m1() const { return stubDR_.z(); }
      // stub projected phi uncertainty in rad
      double d0() const { return stubDR_.dPhi(); }
      // stub projected z uncertainty in cm
      double d1() const { return stubDR_.dZ(); }
      // squared stub projected phi uncertainty instead of wheight (wrong but simpler)
      double v0() const { return v0_; }
      // squared stub projected z uncertainty instead of wheight (wrong but simpler)
      double v1() const { return v1_; }
      StubDR stubDR_;
      double v0_;
      double v1_;
    };
    // copy constructor
    State(State*);
    // proto state constructor
    State(KalmanFilterFormats*, TrackDR*, int, const std::vector<Stub*>&, const std::vector<Stub*>&);
    // updated state constructor
    State(State*, const std::vector<double>&);
    // combinatoric state constructor
    State(State*, State*, int);
    ~State() = default;
    //
    State* comb(std::deque<State>&, int);
    //
    State* final(std::deque<State>&, int);
    // input track
    TrackDR* trackDR() const { return trackDR_; }
    // parent state (nullpointer if no parent available)
    State* parent() const { return parent_; }
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
    // proj Stub for given kf layer [0-7]
    Stub* proj(int layer) const { return proj_[layer]; }
    // seed Stub for given seeding layer [0-1]
    Stub* seed(int layer) const { return seed_[layer]; }

  private:
    // provides data fomats
    KalmanFilterFormats* kff_;
    // provides run-time constants
    const Setup* setup_;
    // input track
    TrackDR* trackDR_;
    // input track seed stubs
    std::vector<Stub*> seed_;
    // input track projection stubs
    std::vector<Stub*> proj_;
    // track id
    int trackId_;
    // previous state, nullptr for first states
    State* parent_;
    // shows which proj layer has been added so far
    TTBV hitPattern_;
    // shows which proj layer the found track has stubs on
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
