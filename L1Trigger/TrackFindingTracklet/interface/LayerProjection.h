#ifndef L1Trigger_TrackFindingTracklet_interface_LayerProjection_h
#define L1Trigger_TrackFindingTracklet_interface_LayerProjection_h

#include "L1Trigger/TrackFindingTracklet/interface/FPGAWord.h"

namespace trklet {

  class Settings;

  class LayerProjection {
  public:
    LayerProjection() { valid_ = false; }

    ~LayerProjection() = default;

    void init(const Settings* settings,
              int projlayer,
              double rproj,
              int iphiproj,
              int izproj,
              int iphider,
              int izder,
              double phiproj,
              double zproj,
              double phiprojder,
              double zprojder,
              double phiprojapprox,
              double zprojapprox,
              double phiprojderapprox,
              double zprojderapprox);

    bool valid() const { return valid_; }

    int projlayer() const {
      assert(valid_);
      return projlayer_;
    };

    double rproj() const {
      assert(valid_);
      return rproj_;
    };

    const FPGAWord& fpgaphiproj() const {
      assert(valid_);
      return fpgaphiproj_;
    };

    const FPGAWord& fpgazproj() const {
      assert(valid_);
      return fpgazproj_;
    };

    const FPGAWord& fpgaphiprojder() const {
      assert(valid_);
      return fpgaphiprojder_;
    };

    const FPGAWord& fpgazprojder() const {
      assert(valid_);
      return fpgazprojder_;
    };

    const FPGAWord& fpgaphiprojvm() const {
      assert(valid_);
      return fpgaphiprojvm_;
    };

    const FPGAWord& fpgazbin1projvm() const {
      assert(valid_);
      return fpgazbin1projvm_;
    };

    const FPGAWord& fpgazbin2projvm() const {
      assert(valid_);
      return fpgazbin2projvm_;
    };

    const FPGAWord& fpgafinezvm() const {
      assert(valid_);
      return fpgafinezvm_;
    };

    const FPGAWord& fpgazprojvm() const {
      assert(valid_);
      return fpgazprojvm_;
    };

    double phiproj() const {
      assert(valid_);
      return phiproj_;
    };

    double zproj() const {
      assert(valid_);
      return zproj_;
    };

    double phiprojder() const {
      assert(valid_);
      return phiprojder_;
    };

    double zprojder() const {
      assert(valid_);
      return zprojder_;
    };

    double phiprojapprox() const {
      assert(valid_);
      return phiprojapprox_;
    };

    double zprojapprox() const {
      assert(valid_);
      return zprojapprox_;
    };

    double phiprojderapprox() const {
      assert(valid_);
      return phiprojderapprox_;
    };

    double zprojderapprox() const {
      assert(valid_);
      return zprojderapprox_;
    };

  protected:
    bool valid_;

    int projlayer_;

    double rproj_;

    FPGAWord fpgaphiproj_;
    FPGAWord fpgazproj_;
    FPGAWord fpgaphiprojder_;
    FPGAWord fpgazprojder_;

    FPGAWord fpgaphiprojvm_;
    FPGAWord fpgazprojvm_;

    FPGAWord fpgazbin1projvm_;
    FPGAWord fpgazbin2projvm_;
    FPGAWord fpgafinezvm_;

    double phiproj_;
    double zproj_;
    double phiprojder_;
    double zprojder_;

    double zbin1_;
    double zbin2_;

    double phiprojapprox_;
    double zprojapprox_;
    double phiprojderapprox_;
    double zprojderapprox_;

    const Settings* settings_;
  };
};  // namespace trklet
#endif
