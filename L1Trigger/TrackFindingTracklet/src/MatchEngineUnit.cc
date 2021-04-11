#include "L1Trigger/TrackFindingTracklet/interface/MatchEngineUnit.h"

using namespace std;
using namespace trklet;

MatchEngineUnit::MatchEngineUnit(bool barrel, unsigned int layerdisk, vector<bool> table) : candmatches_(5) {
  idle_ = true;
  barrel_ = barrel;
  table_ = table;
  layerdisk_ = layerdisk;
}

void MatchEngineUnit::init(VMStubsMEMemory* vmstubsmemory,
                           unsigned int nrzbins,
                           unsigned int rzbin,
                           unsigned int phibin,
                           int shift,
                           int projrinv,
                           int projfinerz,
                           int projfinephi,
                           bool usefirstMinus,
                           bool usefirstPlus,
                           bool usesecondMinus,
                           bool usesecondPlus,
                           bool isPSseed,
                           Tracklet* proj) {
  vmstubsmemory_ = vmstubsmemory;
  idle_ = false;
  nrzbins_ = nrzbins;
  rzbin_ = rzbin;
  phibin_ = phibin;
  shift_ = shift;
  istub_ = 0;
  iuse_ = 0;
  projrinv_ = projrinv;
  projfinerz_ = projfinerz;
  projfinephi_ = projfinephi;
  use_.clear();
  if (usefirstMinus) {
    use_.emplace_back(0, 0);
  }
  if (usefirstPlus) {
    use_.emplace_back(0, 1);
  }
  if (usesecondMinus) {
    use_.emplace_back(1, 0);
  }
  if (usesecondPlus) {
    use_.emplace_back(1, 1);
  }
  assert(use_.size() != 0);
  isPSseed_ = isPSseed;
  proj_ = proj;
}

void MatchEngineUnit::step(bool print) {
  if (idle() || candmatches_.almostfull())
    return;

  unsigned int slot = (phibin_ + use_[iuse_].second) * nrzbins_ + rzbin_ + use_[iuse_].first;

  int projfinerz = projfinerz_ - (1 << NFINERZBITS) * use_[iuse_].first;
  int projfinephi = projfinephi_;
  if (use_[iuse_].second == 0) {
    if (shift_ == -1) {
      projfinephi -= (1 << NFINEPHIBITS);
    }
  } else {
    //When we get here shift_ is either 1 or -1
    if (shift_ == 1) {
      projfinephi += (1 << NFINEPHIBITS);
    }
  }

  const VMStubME& vmstub = vmstubsmemory_->getVMStubMEBin(slot, istub_);

  bool isPSmodule = vmstub.isPSmodule();
  int stubfinerz = vmstub.finerz().value();
  int stubfinephi = vmstub.finephi().value();

  int deltaphi = stubfinephi - projfinephi;

  bool dphicut = (abs(deltaphi) < 3);

  int nbits = isPSmodule ? 3 : 4;

  int diskps = (!barrel_) && isPSmodule;

  unsigned int index = (diskps << (4 + 5)) + (projrinv_ << nbits) + vmstub.bend().value();

  //Check if stub z position consistent
  int idrz = stubfinerz - projfinerz;
  bool pass;

  if (barrel_) {
    if (isPSseed_) {
      pass = idrz >= -1 && idrz <= 1;
    } else {
      pass = idrz >= -5 && idrz <= 5;
    }
  } else {
    if (isPSmodule) {
      pass = idrz >= -1 && idrz <= 1;
    } else {
      pass = idrz >= -3 && idrz <= 3;
    }
  }

  if (print)
    cout << "MEU TrkId stubindex : " << 128 * proj_->TCIndex() + proj_->trackletIndex() << " "
         << vmstub.stubindex().value() << "   " << ((pass && dphicut) && table_[index]) << " index=" << index
         << " projrinv bend : " << projrinv_ << " " << vmstub.bend().value() << "  shift_ isPSseed_ :" << shift_ << " "
         << isPSseed_ << " slot=" << slot << endl;

  //Check if stub bend and proj rinv consistent
  if ((pass && dphicut) && table_[index]) {
    std::pair<Tracklet*, const Stub*> tmp(proj_, vmstub.stub());
    candmatches_.store(tmp);
  }

  istub_++;
  if (istub_ >= vmstubsmemory_->nStubsBin(slot)) {
    iuse_++;
    if (iuse_ < use_.size()) {
      istub_ = 0;
    } else {
      idle_ = true;
    }
  }
}

void MatchEngineUnit::reset() {
  candmatches_.reset();
  idle_ = true;
  istub_ = 0;
}
