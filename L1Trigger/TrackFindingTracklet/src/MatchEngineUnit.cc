#include "L1Trigger/TrackFindingTracklet/interface/MatchEngineUnit.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletLUT.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

using namespace std;
using namespace trklet;

MatchEngineUnit::MatchEngineUnit(const Settings& settings,
                                 bool barrel,
                                 unsigned int layerdisk,
                                 const TrackletLUT& luttable)
    : settings_(settings),
      isPSseed_(false),
      idle_(true),
      almostfullsave_(false),
      luttable_(luttable),
      isPSseed__(false),
      isPSseed___(false),
      isPSseed____(false),
      good__(false),
      good___(false),
      good____(false),
      candmatches_(3),
      print_(false) {
  imeu_ = -1;
  barrel_ = barrel;
  layerdisk_ = layerdisk;
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
                           Tracklet* proj, bool print) {
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

  if (print) {
    std::cout << "MEU Init: " << imeu_ << " " << projfinerz << " " << projfinephi << std::endl;
  }
  
  good__ = false;
}

void MatchEngineUnit::step(bool print) {
  good__ = !idle() && !almostfullsave_;

  if (!good__)
    return;

  unsigned int slot = (phibin_ + use_[iuse_].second) * nrzbins_ + rzbin_ + use_[iuse_].first;

  projfinerz__ = projfinerz_ - (1 << NFINERZBITS) * use_[iuse_].first;
  projfinephi__ = projfinephi_;
  if (use_[iuse_].second == 0) {
    if (shift_ == -1) {
      projfinephi__ -= (1 << NFINEPHIBITS);
    }
  } else {
    //When we get here shift_ is either 1 or -1
    if (shift_ == 1) {
      projfinephi__ += (1 << NFINEPHIBITS);
    }
  }

  vmstub__ = vmstubsmemory_->getVMStubMEBin(slot, istub_);

  if (print) {
    std::cout << "Read vmstub MEU "<<imeu_<<" "<<slot<<" "<<istub_<<" "<<vmstub__.bend().value() << std::endl;
  }

  rzbin__ = rzbin_ + use_[iuse_].first;

  isPSseed__ = isPSseed_;
  projrinv__ = projrinv_;
  proj__ = proj_;

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

void MatchEngineUnit::processPipeline(bool print) {
  if (good____) {
    int stubfinerz = vmstub____.finerz().value();
    int stubfinephi = vmstub____.finephi().value();
    bool isPSmodule = false;

    if (barrel_) {
      isPSmodule = layerdisk_ < N_PSLAYER;
    } else {
      const int absz = (1 << settings_.MEBinsBits()) - 1;
      unsigned int irstub = ((rzbin____ & absz) << NFINERZBITS) + stubfinerz;

      //Verify that ir2smin_ is initialized and check if irstub is less than radius of innermost 2s module
      assert(ir2smin_ > 0);
      isPSmodule = irstub < ir2smin_;
    }
    assert(isPSmodule == vmstub____.isPSmodule());

    int deltaphi = stubfinephi - projfinephi____;

    constexpr int idphicut = 3;

    bool dphicut = (abs(deltaphi) < idphicut);

    int nbits = isPSmodule ? N_BENDBITS_PS : N_BENDBITS_2S;

    int diskps = (!barrel_) && isPSmodule;

    //here we always use the larger number of bits for the bend

    //unsigned int index = (diskps << (N_BENDBITS_2S + NRINVBITS)) + (projrinv___ << nbits) + vmstub___.bend().value(); ????

    unsigned int index = (diskps << (N_BENDBITS_2S + NRINVBITS)) + (projrinv____ << nbits) + vmstub____.bend().value();

    //Check if stub z position consistent
    int idrz = stubfinerz - projfinerz____;
    bool pass;

    if (barrel_) {
      if (isPSseed____) {
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

    if (print) {
      std::cout << "MEU: "<< imeu_ << " " << " " << " " << index << " " 
		<< luttable_.lookup(index) << " " << pass << " " << dphicut << " "
		<< stubfinephi << " " << projfinephi____ << std::endl;
    }

    bool goodpair = (pass && dphicut) && luttable_.lookup(index);

    std::pair<Tracklet*, const Stub*> tmppair(proj____, vmstub____.stub());

    if (goodpair) {
      if (print) {
	std::cout << "Write tp match buffer : " << imeu_ << " " <<candmatches_.rptr() << std::endl;
      }
      candmatches_.store(tmppair);
    }
  }

  proj____ = proj__;
  projfinephi____ = projfinephi__;
  projfinerz____ = projfinerz__;
  projrinv____ = projrinv__;
  isPSseed____ = isPSseed__;
  good____ = good__;
  vmstub____ = vmstub__;
  rzbin____ = rzbin__;

}

void MatchEngineUnit::reset() {
  candmatches_.reset();
  idle_ = true;
  istub_ = 0;
  good__ = false;
  good___ = false;
  good____ = false;
}

int MatchEngineUnit::TCID() const {
  if (!empty()) {
    return peek().first->TCID();
  }

  if (good____) {
    return proj____->TCID();
  }

  if (good___) {
    return proj___->TCID();
  }

  if (good__) {
    return proj__->TCID();
  }

  if (idle_) {
    return (1 << (settings_.nbitstrackletindex() + settings_.nbitstcindex())) - 1;
  }

  return proj_->TCID();
}
