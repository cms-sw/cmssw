#include "L1Trigger/TrackFindingTracklet/interface/MatchEngineUnit.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletLUT.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

using namespace std;
using namespace trklet;

MatchEngineUnit::MatchEngineUnit(const Settings& settings,
                                 bool barrel,
                                 unsigned int layerdisk,
                                 const TrackletLUT& luttable)
    : settings_(settings), luttable_(luttable), candmatches_(3) {
  idle_ = true;
  print_ = false;
  imeu_ = -1;
  barrel_ = barrel;
  layerdisk_ = layerdisk;
  good_in = false;
  good_pipeline = false;
  good_out = false;
  ir2smin_ = 0;
  if (layerdisk_ >= N_LAYER) {
    double rmin2s = (layerdisk_ < N_LAYER + 2) ? settings_.rDSSinner(0) : settings_.rDSSouter(0);
    ir2smin_ = (1 << (N_RZBITS + NFINERZBITS)) * (rmin2s - settings_.rmindiskvm()) /
               (settings_.rmaxdisk() - settings_.rmindiskvm());
  }
}

void MatchEngineUnit::setAlmostFull() { almostfullsave_ = candmatches_.nearfull(); }

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

  good_in = false;
}

void MatchEngineUnit::step() {
  good_in = !idle() && !almostfullsave_;

  if (!good_in)
    return;

  unsigned int slot = (phibin_ + use_[iuse_].second) * nrzbins_ + rzbin_ + use_[iuse_].first;

  projfinerz_in = projfinerz_ - (1 << NFINERZBITS) * use_[iuse_].first;
  projfinephi_in = projfinephi_;
  if (use_[iuse_].second == 0) {
    if (shift_ == -1) {
      projfinephi_in -= (1 << NFINEPHIBITS);
    }
  } else {
    //When we get here shift_ is either 1 or -1
    if (shift_ == 1) {
      projfinephi_in += (1 << NFINEPHIBITS);
    }
  }

  vmstub_in = vmstubsmemory_->getVMStubMEBin(slot, istub_);
  rzbin_in = rzbin_ + use_[iuse_].first;

  isPSseed_in = isPSseed_;
  projrinv_in = projrinv_;
  proj_in = proj_;

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

void MatchEngineUnit::processPipeline() {
  if (good_out) {
    int stubfinerz = vmstub_out.finerz().value();
    int stubfinephi = vmstub_out.finephi().value();
    bool isPSmodule = false;

    if (barrel_) {
      isPSmodule = layerdisk_ < N_PSLAYER;
    } else {
      const int absz = (1 << settings_.MEBinsBits()) - 1;
      unsigned int irstub = ((rzbin_out & absz) << NFINERZBITS) + stubfinerz;

      //Verify that ir2smin_ is initialized and check if irstub is less than radius of innermost 2s module
      assert(ir2smin_ > 0);
      isPSmodule = irstub < ir2smin_;
    }
    assert(isPSmodule == vmstub_out.isPSmodule());

    int deltaphi = stubfinephi - projfinephi_out;

    constexpr int idphicut = 3;

    bool dphicut = (abs(deltaphi) < idphicut);

    int nbits = isPSmodule ? N_BENDBITS_PS : N_BENDBITS_2S;

    int diskps = (!barrel_) && isPSmodule;

    //here we always use the larger number of bits for the bend
    unsigned int index = (diskps << (nbits + NRINVBITS)) + (projrinv_out << nbits) + vmstub_out.bend().value();

    //Check if stub z position consistent
    int idrz = stubfinerz - projfinerz_out;
    bool pass;

    if (barrel_) {
      if (isPSseed_out) {
        constexpr int drzcut = 1;
        pass = std::abs(idrz) <= drzcut;
      } else {
        constexpr int drzcut = 5;
        pass = std::abs(idrz) <= drzcut;
      }
    } else {
      if (isPSmodule) {
        constexpr int drzcut = 1;
        pass = std::abs(idrz) <= drzcut;
      } else {
        constexpr int drzcut = 3;
        pass = std::abs(idrz) <= drzcut;
      }
    }

    bool goodpair = (pass && dphicut) && luttable_.lookup(index);

    std::pair<Tracklet*, const Stub*> tmppair(proj_out, vmstub_out.stub());

    if (goodpair) {
      candmatches_.store(tmppair);
    }
  }

  proj_out = proj_pipeline;
  projfinephi_out = projfinephi_pipeline;
  projfinerz_out = projfinerz_pipeline;
  projrinv_out = projrinv_pipeline;
  isPSseed_out = isPSseed_pipeline;
  good_out = good_pipeline;
  vmstub_out = vmstub_pipeline;
  rzbin_out = rzbin_pipeline;

  proj_pipeline = proj_in;
  projfinephi_pipeline = projfinephi_in;
  projfinerz_pipeline = projfinerz_in;
  projrinv_pipeline = projrinv_in;
  isPSseed_pipeline = isPSseed_in;
  good_pipeline = good_in;
  vmstub_pipeline = vmstub_in;
  rzbin_pipeline = rzbin_in;
}

void MatchEngineUnit::reset() {
  candmatches_.reset();
  idle_ = true;
  istub_ = 0;
  good_in = false;
  good_pipeline = false;
  good_out = false;
}

int MatchEngineUnit::TCID() const {
  if (!empty()) {
    return peek().first->TCID();
  }

  if (good_out) {
    return proj_out->TCID();
  }

  if (good_pipeline) {
    return proj_pipeline->TCID();
  }

  if (good_in) {
    return proj_in->TCID();
  }

  if (idle_) {
    return (1 << (settings_.nbitstrackletindex() + settings_.nbitstcindex())) - 1;
  }

  return proj_->TCID();
}
