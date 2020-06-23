#ifndef L1Trigger_TrackFindingTracklet_interface_DiskProjection_h
#define L1Trigger_TrackFindingTracklet_interface_DiskProjection_h

#include "L1Trigger/TrackFindingTracklet/interface/FPGAWord.h"

#include <cassert>

namespace trklet {

  class Settings;

  class DiskProjection {
  public:
    DiskProjection() { valid_ = false; }

    ~DiskProjection() = default;

    void init(Settings const& settings,
              int projdisk,
              double zproj,
              int iphiproj,
              int irproj,
              int iphider,
              int irder,
              double phiproj,
              double rproj,
              double phiprojder,
              double rprojder,
              double phiprojapprox,
              double rprojapprox,
              double phiprojderapprox,
              double rprojderapprox);

    bool valid() const { return valid_; }

    int projdisk() const {
      assert(valid_);
      return projdisk_;
    };

    double zproj() const {
      assert(valid_);
      return zproj_;
    };

    const FPGAWord& fpgaphiproj() const {
      assert(valid_);
      return fpgaphiproj_;
    };

    const FPGAWord& fpgarproj() const {
      assert(valid_);
      return fpgarproj_;
    };

    const FPGAWord& fpgaphiprojder() const {
      assert(valid_);
      return fpgaphiprojder_;
    };

    const FPGAWord& fpgarprojder() const {
      assert(valid_);
      return fpgarprojder_;
    };

    const FPGAWord& fpgaphiprojvm() const {
      assert(valid_);
      return fpgaphiprojvm_;
    };

    const FPGAWord& fpgarprojvm() const {
      assert(valid_);
      return fpgarprojvm_;
    };

    double phiproj() const {
      assert(valid_);
      return phiproj_;
    };

    const FPGAWord& fpgarbin1projvm() const {
      assert(valid_);
      return fpgarbin1projvm_;
    };

    const FPGAWord& fpgarbin2projvm() const {
      assert(valid_);
      return fpgarbin2projvm_;
    };

    const FPGAWord& fpgafinervm() const {
      assert(valid_);
      return fpgafinervm_;
    };

    double rproj() const {
      assert(valid_);
      return rproj_;
    };

    double phiprojder() const {
      assert(valid_);
      return phiprojder_;
    };

    double rprojder() const {
      assert(valid_);
      return rprojder_;
    };

    double phiprojapprox() const {
      assert(valid_);
      return phiprojapprox_;
    };

    double rprojapprox() const {
      assert(valid_);
      return rprojapprox_;
    };

    double phiprojderapprox() const {
      assert(valid_);
      return phiprojderapprox_;
    };

    double rprojderapprox() const {
      assert(valid_);
      return rprojderapprox_;
    };

    void setBendIndex(int bendindex) { fpgabendindex_.set(bendindex, 5, true, __LINE__, __FILE__); }

    const FPGAWord& getBendIndex() const { return fpgabendindex_; }

  protected:
    bool valid_;

    int projdisk_;

    double zproj_;

    FPGAWord fpgaphiproj_;
    FPGAWord fpgarproj_;
    FPGAWord fpgaphiprojder_;
    FPGAWord fpgarprojder_;

    FPGAWord fpgaphiprojvm_;
    FPGAWord fpgarprojvm_;

    FPGAWord fpgarbin1projvm_;
    FPGAWord fpgarbin2projvm_;
    FPGAWord fpgafinervm_;

    FPGAWord fpgabendindex_;

    double phiproj_;
    double rproj_;
    double phiprojder_;
    double rprojder_;

    double phiprojapprox_;
    double rprojapprox_;
    double phiprojderapprox_;
    double rprojderapprox_;
  };

};  // namespace trklet
#endif
