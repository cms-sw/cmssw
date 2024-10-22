#ifndef L1Trigger_L1THGCal_HGCalMulticluster_SA_h
#define L1Trigger_L1THGCal_HGCalMulticluster_SA_h

#include "L1Trigger/L1THGCal/interface/backend/HGCalCluster_SA.h"

#include <vector>

namespace l1thgcfirmware {

  class HGCalMulticluster {
  public:
    HGCalMulticluster()
        : centre_x_(0),
          centre_y_(0),
          centre_z_(0),
          centreProj_x_(0),
          centreProj_y_(0),
          centreProj_z_(0),
          mipPt_(0),
          sumPt_() {}

    HGCalMulticluster(const l1thgcfirmware::HGCalCluster& tc, float fraction = 1.);

    void addConstituent(const l1thgcfirmware::HGCalCluster& tc, bool updateCentre = true, float fraction = 1.);

    ~HGCalMulticluster() = default;

    const std::vector<l1thgcfirmware::HGCalCluster>& constituents() const { return constituents_; }

    unsigned size() const { return constituents_.size(); }

    float sumPt() const { return sumPt_; }

  private:
    // Could replace this with own simple implementation of GlobalPoint?
    // Or just a struct?
    float centre_x_;
    float centre_y_;
    float centre_z_;

    float centreProj_x_;
    float centreProj_y_;
    float centreProj_z_;

    float mipPt_;
    float sumPt_;

    std::vector<l1thgcfirmware::HGCalCluster> constituents_;

    void updateP4AndPosition(const l1thgcfirmware::HGCalCluster& tc, bool updateCentre = true, float fraction = 1.);
  };

  typedef std::vector<HGCalMulticluster> HGCalMulticlusterSACollection;

}  // namespace l1thgcfirmware

#endif
