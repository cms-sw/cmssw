#include "L1Trigger/TrackFindingTracklet/interface/ProjectionTemp.h"

using namespace std;
using namespace trklet;

ProjectionTemp::ProjectionTemp(Tracklet* proj,
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
                               bool isPSseed) {
  proj_ = proj;
  slot_ = slot;
  projrinv_ = projrinv;
  projfinerz_ = projfinerz;
  projfinephi_ = projfinephi;
  iphi_ = iphi;
  shift_ = shift;
  use_[0][0] = usefirstMinus;
  use_[0][1] = usefirstPlus;
  use_[1][0] = usesecondMinus;
  use_[1][1] = usesecondPlus;
  isPSseed_ = isPSseed;
}

ProjectionTemp::ProjectionTemp() {
  proj_ = nullptr;
  slot_ = 0;
  projrinv_ = 0;
  projfinerz_ = 0;
  projfinephi_ = 0;
  iphi_ = 0;
  shift_ = 0;
  use_[0][0] = false;
  use_[0][1] = false;
  use_[1][0] = false;
  use_[1][1] = false;
  isPSseed_ = false;
}
