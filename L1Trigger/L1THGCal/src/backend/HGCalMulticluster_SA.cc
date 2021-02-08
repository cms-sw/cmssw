#include "L1Trigger/L1THGCal/interface/backend/HGCalMulticluster_SA.h"

#include <cmath>

using namespace l1thgcfirmware;

HGCalMulticluster::HGCalMulticluster(const HGCalCluster& tc, float fraction) {
  HGCalMulticluster();
  addConstituent(tc, true, fraction);
}

void HGCalMulticluster::addConstituent(const HGCalCluster& tc, bool updateCentre, float fraction) {
  // If no constituents, set seedMiptPt to cluster mipPt
  if (constituents_.empty()) {
    // seedMipPt_ = cMipPt;
    if (!updateCentre) {
      centre_x_ = tc.x();
      centre_y_ = tc.y();
      centre_z_ = tc.z();
    }
  }
  // UpdateP4AndPosition
  updateP4AndPosition(tc, updateCentre, fraction);

  constituents_.emplace_back(tc);
}

void HGCalMulticluster::updateP4AndPosition(const HGCalCluster& tc, bool updateCentre, float fraction) {
  // Get cluster mipPt
  double cMipt = tc.mipPt() * fraction;
  double cPt = tc.pt() * fraction;
  if (updateCentre) {
    float clusterCentre_x = centre_x_ * mipPt_ + tc.x() * cMipt;
    float clusterCentre_y = centre_y_ * mipPt_ + tc.y() * cMipt;
    float clusterCentre_z = centre_z_ * mipPt_ + tc.z() * cMipt;  // Check this!

    if ((mipPt_ + cMipt) > 0) {
      clusterCentre_x /= (mipPt_ + cMipt);
      clusterCentre_y /= (mipPt_ + cMipt);
      clusterCentre_z /= (mipPt_ + cMipt);
    }
    centre_x_ = clusterCentre_x;
    centre_y_ = clusterCentre_y;
    centre_z_ = clusterCentre_z;

    if (centre_z_ != 0) {
      centreProj_x_ = centre_x_ / std::abs(centre_z_);
      centreProj_y_ = centre_y_ / std::abs(centre_z_);
      centreProj_z_ = centre_z_ / std::abs(centre_z_);
    }
  }

  mipPt_ += cMipt;
  sumPt_ += cPt;
}