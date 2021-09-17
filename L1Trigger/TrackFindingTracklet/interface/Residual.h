#ifndef L1Trigger_TrackFindingTracklet_interface_Residual_h
#define L1Trigger_TrackFindingTracklet_interface_Residual_h

#include "L1Trigger/TrackFindingTracklet/interface/FPGAWord.h"

#include <cassert>

namespace trklet {

  class Settings;
  class Stub;

  class Residual {
  public:
    Residual() { valid_ = false; }

    ~Residual() = default;

    void init(Settings const& settings,
              unsigned int layerdisk,
              int iphiresid,
              int irzresid,
              int istubid,
              double phiresid,
              double rzresid,
              double phiresidapprox,
              double rzresidapprox,
              const Stub* stubptr);

    bool valid() const { return valid_; }

    const FPGAWord& fpgaphiresid() const {
      assert(valid_);
      return fpgaphiresid_;
    };

    const FPGAWord& fpgarzresid() const {
      assert(valid_);
      return fpgarzresid_;
    };

    const FPGAWord& fpgastubid() const {
      assert(valid_);
      return fpgastubid_;
    };

    double phiresid() const {
      assert(valid_);
      return phiresid_;
    };

    double rzresid() const {
      assert(valid_);
      return rzresid_;
    };

    double phiresidapprox() const {
      assert(valid_);
      return phiresidapprox_;
    };

    double rzresidapprox() const {
      assert(valid_);
      return rzresidapprox_;
    };

    const Stub* stubptr() const {
      assert(valid_);
      return stubptr_;
    }

  protected:
    bool valid_;

    unsigned int layerdisk_;

    FPGAWord fpgaphiresid_;
    FPGAWord fpgarzresid_;
    FPGAWord fpgastubid_;

    double phiresid_;
    double rzresid_;

    double phiresidapprox_;
    double rzresidapprox_;

    const Stub* stubptr_;
  };

};  // namespace trklet
#endif
