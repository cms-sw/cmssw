#include "L1Trigger/TrackFindingTracklet/interface/MatchEngineUnit.h"

using namespace std;
using namespace trklet;

MatchEngineUnit::MatchEngineUnit(bool barrel, vector<bool> table, vector<bool> tablePS, vector<bool> table2S)
    : candmatches_(5) {
  idle_ = true;
  barrel_ = barrel;
  table_ = table;
  tablePS_ = tablePS;
  table2S_ = table2S;
  slot_ = 1;  //This makes it idle until initialized
}

void MatchEngineUnit::init(VMStubsMEMemory* vmstubsmemory,
                           unsigned int slot,
                           int projrinv,
                           int projfinerz,
                           int projfinephi,
                           bool isPSseed,
                           Tracklet* proj) {
  vmstubsmemory_ = vmstubsmemory;
  idle_ = false;
  slot_ = slot;
  istub_ = 0;
  projrinv_ = projrinv;
  projfinerz_ = projfinerz;
  projfinephi_ = projfinephi;
  isPSseed_ = isPSseed;
  proj_ = proj;
}

void MatchEngineUnit::step() {
  if (idle() || candmatches_.almostfull())
    return;

  const VMStubME& vmstub = vmstubsmemory_->getVMStubMEBin(slot_, istub_);

  istub_++;
  if (istub_ >= vmstubsmemory_->nStubsBin(slot_))
    idle_ = true;

  bool isPSmodule = vmstub.isPSmodule();
  int stubfinerz = vmstub.finerz().value();
  int stubfinephi = vmstub.finephi().value();

  int deltaphi = stubfinephi - projfinephi_;

  bool dphicut = (abs(deltaphi) < 3) || (abs(deltaphi) > 5);  //TODO - need better implementations

  if (!barrel_)
    dphicut = true;

  int nbits = isPSmodule ? 3 : 4;

  unsigned int index = (projrinv_ << nbits) + vmstub.bend().value();

  //Check if stub z position consistent
  int idrz = stubfinerz - projfinerz_;
  bool pass;

  if (barrel_) {
    if (isPSseed_) {
      pass = idrz >= -2 && idrz <= 2;
    } else {
      pass = idrz >= -5 && idrz <= 5;
    }
  } else {
    if (isPSmodule) {
      pass = idrz >= -1 && idrz <= 1;
    } else {
      pass = idrz >= -5 && idrz <= 5;
    }
  }

  //Check if stub bend and proj rinv consistent
  if ((pass && dphicut) && (barrel_ ? table_[index] : (isPSmodule ? tablePS_[index] : table2S_[index]))) {
    std::pair<Tracklet*, const Stub*> tmp(proj_, vmstub.stub());
    candmatches_.store(tmp);
  }
}
