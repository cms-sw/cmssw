#include "L1Trigger/TrackFindingTracklet/interface/MatchEngineUnit.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletLUT.h"

using namespace std;
using namespace trklet;

MatchEngineUnit::MatchEngineUnit(bool barrel, unsigned int layerdisk, const TrackletLUT& luttable)
    : luttable_(luttable), candmatches_(3) {
  idle_ = true;
  barrel_ = barrel;
  layerdisk_ = layerdisk;
  goodpair_ = false;
  goodpair__ = false;
  havepair_ = false;
  havepair__ = false;
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
                           Tracklet* proj,
                           bool) {
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
  if (usesecondMinus) {
    use_.emplace_back(1, 0);
  }
  if (usefirstPlus) {
    use_.emplace_back(0, 1);
  }
  if (usesecondPlus) {
    use_.emplace_back(1, 1);
  }
  assert(!use_.empty());
  isPSseed_ = isPSseed;
  proj_ = proj;

  //Even when you init a new projection you need to process the pipeline
  //This should be fixed to be done more cleanly - but require synchronizaton
  //with the HLS code
  if (goodpair__) {
    candmatches_.store(tmppair__);
  }

  havepair__ = havepair_;
  goodpair__ = goodpair_;
  tmppair__ = tmppair_;

  havepair_ = false;
  goodpair_ = false;
}

void MatchEngineUnit::step(bool) {
  bool almostfull = candmatches_.nearfull();

  if (goodpair__) {
    assert(havepair__);
    candmatches_.store(tmppair__);
  }

  havepair__ = havepair_;
  goodpair__ = goodpair_;
  tmppair__ = tmppair_;

  havepair_ = false;
  goodpair_ = false;

  if (idle() || almostfull)
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

  // Detailed printout for comparison with HLS code
  bool print = false;
  if (print)
    edm::LogVerbatim("Tracklet") << "MEU TrkId stubindex : " << 128 * proj_->TCIndex() + proj_->trackletIndex() << " "
                                 << vmstub.stubindex().value() << "   "
                                 << ((pass && dphicut) && luttable_.lookup(index)) << " index=" << index
                                 << " projrinv bend : " << projrinv_ << " " << vmstub.bend().value()
                                 << "  shift_ isPSseed_ :" << shift_ << " " << isPSseed_ << " slot=" << slot;

  //Check if stub bend and proj rinv consistent

  goodpair_ = (pass && dphicut) && luttable_.lookup(index);
  havepair_ = true;

  if (havepair_) {
    std::pair<Tracklet*, const Stub*> tmppair(proj_, vmstub.stub());
    tmppair_ = tmppair;
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
  goodpair_ = false;
  goodpair__ = false;
  havepair_ = false;
  havepair__ = false;
}

int MatchEngineUnit::TCID() const {
  if (!empty()) {
    return peek().first->TCID();
  }

  if (idle_ && !havepair_ && !havepair__) {
    return 16383;
  }
  if (havepair__) {
    return tmppair__.first->TCID();
  }
  if (havepair_) {
    return tmppair_.first->TCID();
  }
  return proj_->TCID();
}
