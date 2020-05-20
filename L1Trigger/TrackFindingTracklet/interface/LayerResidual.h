#ifndef L1Trigger_TrackFindingTracklet_interface_LayerResidual_h
#define L1Trigger_TrackFindingTracklet_interface_LayerResidual_h

#include "L1Trigger/TrackFindingTracklet/interface/FPGAWord.h"

#include <cassert>

namespace trklet {

  class Settings;
  class Stub;

  class LayerResidual {
  public:
    LayerResidual() { valid_ = false; }

    ~LayerResidual() = default;

    void init(const Settings* settings,
              int layer,
              int iphiresid,
              int izresid,
              int istubid,
              double phiresid,
              double zresid,
              double phiresidapprox,
              double zresidapprox,
              double rstub,
              const Stub* stubptr);

    bool valid() const { return valid_; }

    const FPGAWord& fpgaphiresid() const {
      assert(valid_);
      return fpgaphiresid_;
    };

    const FPGAWord& fpgazresid() const {
      assert(valid_);
      return fpgazresid_;
    };

    const FPGAWord& fpgastubid() const {
      assert(valid_);
      return fpgastubid_;
    };

    double phiresid() const {
      assert(valid_);
      return phiresid_;
    };

    double zresid() const {
      assert(valid_);
      return zresid_;
    };

    double phiresidapprox() const {
      assert(valid_);
      return phiresidapprox_;
    };

    double zresidapprox() const {
      assert(valid_);
      return zresidapprox_;
    };

    double rstub() const {
      assert(valid_);
      return rstub_;
    }

    const Stub* stubptr() const {
      assert(valid_);
      return stubptr_;
    }

  protected:
    bool valid_;

    int layer_;

    FPGAWord fpgaphiresid_;
    FPGAWord fpgazresid_;
    FPGAWord fpgastubid_;

    double phiresid_;
    double zresid_;

    double phiresidapprox_;
    double zresidapprox_;

    double rstub_;
    const Stub* stubptr_;
  };

};  // namespace trklet
#endif
