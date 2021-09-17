#ifndef L1Trigger_TrackFindingTracklet_interface_ProjectionTemp_h
#define L1Trigger_TrackFindingTracklet_interface_ProjectionTemp_h

#include <cassert>
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"

namespace trklet {

  class ProjectionTemp {
  public:
    ProjectionTemp(Tracklet* proj,
                   unsigned int slot,
                   unsigned int projrinv,
                   int projfinerz,
                   unsigned int projfinephi,
                   unsigned int iphi,
                   int shift,
                   bool usefirstMinus,
                   bool usefirstPlus,
                   bool usesecondMinus,
                   bool usesecondPlus,
                   bool isPSseed);

    ProjectionTemp();

    ~ProjectionTemp() = default;

    Tracklet* proj() const { return proj_; }
    unsigned int slot() const { return slot_; }
    unsigned int projrinv() const { return projrinv_; }
    int projfinerz() const { return projfinerz_; }
    unsigned int projfinephi() const { return projfinephi_; }
    unsigned int iphi() const { return iphi_; }
    int shift() const { return shift_; }
    bool use(unsigned int nextrzbin, unsigned int nextiphibin) const { return use_[nextrzbin][nextiphibin]; }
    bool isPSseed() const { return isPSseed_; }

  private:
    Tracklet* proj_;
    unsigned int slot_;
    unsigned int projrinv_;
    unsigned int projfinerz_;
    unsigned int projfinephi_;
    unsigned int iphi_;
    int shift_;
    //Projection may use two bins in rz and phi if the projection is near a boundary
    //The use_[rz][phi] array indicates which bins are used.
    bool use_[2][2];
    bool isPSseed_;
  };
};  // namespace trklet
#endif
