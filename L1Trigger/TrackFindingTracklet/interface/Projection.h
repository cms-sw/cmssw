#ifndef L1Trigger_TrackFindingTracklet_interface_Projection_h
#define L1Trigger_TrackFindingTracklet_interface_Projection_h

#include "L1Trigger/TrackFindingTracklet/interface/FPGAWord.h"

namespace trklet {

  class Settings;

  class Projection {
  public:
    Projection() { valid_ = false; }

    ~Projection() = default;

    void init(Settings const& settings,
              unsigned int layerdisk,
              int iphiproj,
              int irzproj,
              int iphider,
              int irzder,
              double phiproj,
              double rzproj,
              double phiprojder,
              double rzprojder,
              double phiprojapprox,
              double rzprojapprox,
              double phiprojderapprox,
              double rzprojderapprox,
              bool isPSseed);

    bool valid() const { return valid_; }

    unsigned int layerdisk() const {
      assert(valid_);
      return layerdisk_;
    };

    const FPGAWord& fpgaphiproj() const {
      assert(valid_);
      return fpgaphiproj_;
    };

    const FPGAWord& fpgarzproj() const {
      assert(valid_);
      return fpgarzproj_;
    };

    const FPGAWord& fpgaphiprojder() const {
      assert(valid_);
      return fpgaphiprojder_;
    };

    const FPGAWord& fpgarzprojder() const {
      assert(valid_);
      return fpgarzprojder_;
    };

    const FPGAWord& fpgarzbin1projvm() const {
      assert(valid_);
      return fpgarzbin1projvm_;
    };

    const FPGAWord& fpgarzbin2projvm() const {
      assert(valid_);
      return fpgarzbin2projvm_;
    };

    const FPGAWord& fpgafinerzvm() const {
      assert(valid_);
      return fpgafinerzvm_;
    };

    const FPGAWord& fpgafinephivm() const {
      assert(valid_);
      return fpgafinephivm_;
    };

    double phiproj() const {
      assert(valid_);
      return phiproj_;
    };

    double rzproj() const {
      assert(valid_);
      return rzproj_;
    };

    double phiprojder() const {
      assert(valid_);
      return phiprojder_;
    };

    double rzprojder() const {
      assert(valid_);
      return rzprojder_;
    };

    double phiprojapprox() const {
      assert(valid_);
      return phiprojapprox_;
    };

    double rzprojapprox() const {
      assert(valid_);
      return rzprojapprox_;
    };

    double phiprojderapprox() const {
      assert(valid_);
      return phiprojderapprox_;
    };

    double rzprojderapprox() const {
      assert(valid_);
      return rzprojderapprox_;
    };

    void setBendIndex(int bendindex) { fpgabendindex_.set(bendindex, 5, true, __LINE__, __FILE__); }

    const FPGAWord& getBendIndex() const { return fpgabendindex_; }

  protected:
    bool valid_;

    unsigned int layerdisk_;

    FPGAWord fpgaphiproj_;
    FPGAWord fpgarzproj_;
    FPGAWord fpgaphiprojder_;
    FPGAWord fpgarzprojder_;

    FPGAWord fpgarzbin1projvm_;
    FPGAWord fpgarzbin2projvm_;
    FPGAWord fpgafinerzvm_;
    FPGAWord fpgafinephivm_;

    double phiproj_;
    double rzproj_;
    double phiprojder_;
    double rzprojder_;

    double phiprojapprox_;
    double rzprojapprox_;
    double phiprojderapprox_;
    double rzprojderapprox_;

    //used by projections to disks
    FPGAWord fpgabendindex_;
  };
};  // namespace trklet
#endif
