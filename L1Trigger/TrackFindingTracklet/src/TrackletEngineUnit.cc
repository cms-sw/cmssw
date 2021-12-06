#include "L1Trigger/TrackFindingTracklet/interface/TrackletEngineUnit.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubsTEMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

using namespace std;
using namespace trklet;

TrackletEngineUnit::TrackletEngineUnit(const Settings* const settings,
                                       unsigned int nbitsfinephi,
                                       unsigned int layerdisk1,
                                       unsigned int layerdisk2,
                                       unsigned int iSeed,
                                       unsigned int nbitsfinephidiff,
                                       unsigned int iAllStub,
                                       const TrackletLUT* pttableinnernew,
                                       const TrackletLUT* pttableouternew,
                                       VMStubsTEMemory* outervmstubs)
    : settings_(settings), pttableinnernew_(pttableinnernew), pttableouternew_(pttableouternew), candpairs_(3) {
  idle_ = true;
  nbitsfinephi_ = nbitsfinephi;
  layerdisk2_ = layerdisk2;
  layerdisk1_ = layerdisk1;
  iSeed_ = iSeed;
  nbitsfinephidiff_ = nbitsfinephidiff;
  iAllStub_ = iAllStub;
  outervmstubs_ = outervmstubs;
}

void TrackletEngineUnit::init(const TEData& tedata) {
  tedata_ = tedata;
  nreg_ = 0;
  istub_ = 0;
  idle_ = false;
  assert(!tedata_.regions_.empty());
  std::tie(next_, ireg_, nstub_) = tedata_.regions_[0];
}

void TrackletEngineUnit::reset() {
  idle_ = true;
  goodpair_ = false;
  goodpair__ = false;
  candpairs_.reset();
}

void TrackletEngineUnit::step(bool, int, int) {
  if (goodpair__) {
    candpairs_.store(candpair__);
  }

  goodpair__ = goodpair_;
  candpair__ = candpair_;

  goodpair_ = false;

  if (idle_ || nearfull_) {
    return;
  }

  int ibin = tedata_.start_ + next_;

  int nbins = (1 << NFINERZBITS);

  assert(istub_ < outervmstubs_->nVMStubsBinned(ireg_ * nbins + ibin));

  const VMStubTE& outervmstub = outervmstubs_->getVMStubTEBinned(ireg_ * nbins + ibin, istub_);
  int rzbin = (outervmstub.vmbits().value() & (nbins - 1));

  FPGAWord iphiouterbin = outervmstub.finephi();

  assert(iphiouterbin == outervmstub.finephi());

  //New code to calculate lut value
  int outerfinephi = iAllStub_ * (1 << (nbitsfinephi_ - settings_->nbitsallstubs(layerdisk2_))) +
                     ireg_ * (1 << settings_->nfinephi(1, iSeed_)) + iphiouterbin.value();
  int idphi = outerfinephi - tedata_.innerfinephi_;

  bool inrange = (idphi < (1 << (nbitsfinephidiff_ - 1))) && (idphi >= -(1 << (nbitsfinephidiff_ - 1)));

  idphi = idphi & ((1 << nbitsfinephidiff_) - 1);

  unsigned int firstDiskSeed = 4;
  if (iSeed_ >= firstDiskSeed) {  //Also use r-position
    int ibinMask = 3;             //Get two least sign. bits
    int ir = ((ibin & ibinMask) << 1) + (rzbin >> (NFINERZBITS - 1));
    int nrbits = 3;
    idphi = (idphi << nrbits) + ir;
  }

  if (next_ != 0)
    rzbin += (1 << NFINERZBITS);

  if ((rzbin < tedata_.rzbinfirst_) || (rzbin - tedata_.rzbinfirst_ > tedata_.rzdiffmax_)) {
    if (settings_->debugTracklet()) {
      edm::LogVerbatim("Tracklet") << " layer-disk stub pair rejected because rbin cut : " << rzbin << " "
                                   << tedata_.rzbinfirst_ << " " << tedata_.rzdiffmax_;
    }
  } else {
    FPGAWord outerbend = outervmstub.bend();

    int ptinnerindex = (idphi << tedata_.innerbend_.nbits()) + tedata_.innerbend_.value();
    int ptouterindex = (idphi << outerbend.nbits()) + outerbend.value();

    if (!(inrange && pttableinnernew_->lookup(ptinnerindex) && pttableouternew_->lookup(ptouterindex))) {
      if (settings_->debugTracklet()) {
        edm::LogVerbatim("Tracklet") << " Stub pair rejected because of stub pt cut bends : "
                                     << settings_->benddecode(
                                            tedata_.innerbend_.value(), layerdisk1_, tedata_.stub_->isPSmodule())
                                     << " "
                                     << settings_->benddecode(outerbend.value(), layerdisk2_, outervmstub.isPSmodule());
      }
    } else {
      candpair_ = pair<const Stub*, const Stub*>(tedata_.stub_, outervmstub.stub());
      goodpair_ = true;
    }
  }

  istub_++;
  assert(nstub_ <= N_VMSTUBSMAX);
  if (istub_ >= nstub_) {
    istub_ = 0;
    nreg_++;
    if (nreg_ >= tedata_.regions_.size()) {
      nreg_ = 0;
      idle_ = true;
    } else {
      std::tie(next_, ireg_, nstub_) = tedata_.regions_[nreg_];
    }
  }
}
