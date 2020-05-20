#include "L1Trigger/TrackFindingTracklet/interface/ProjectionTemp.h"

using namespace std;
using namespace trklet;

ProjectionTemp::ProjectionTemp(Tracklet* proj,
                               unsigned int slot,
                               unsigned int projrinv,
                               int projfinerz,
                               unsigned int projfinephi,
                               unsigned int iphi,
                               bool isPSseed) {
  proj_ = proj;
  slot_ = slot;
  projrinv_ = projrinv;
  projfinerz_ = projfinerz;
  projfinephi_ = projfinephi;
  iphi_ = iphi;
  isPSseed_ = isPSseed;
}

ProjectionTemp::ProjectionTemp() {
  proj_ = nullptr;
  slot_ = 0;
  projrinv_ = 0;
  projfinerz_ = 0;
  projfinephi_ = 0;
  iphi_ = 0;
  isPSseed_ = false;
}
