#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include "L1Trigger/TrackFindingTracklet/interface/Track.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>

#include <sstream>

using namespace std;
using namespace trklet;

Tracklet::Tracklet(Settings const& settings,
                   unsigned int iSeed,
                   const Stub* innerFPGAStub,
                   const Stub* middleFPGAStub,
                   const Stub* outerFPGAStub,
                   double rinv,
                   double phi0,
                   double d0,
                   double z0,
                   double t,
                   double rinvapprox,
                   double phi0approx,
                   double d0approx,
                   double z0approx,
                   double tapprox,
                   int irinv,
                   int iphi0,
                   int id0,
                   int iz0,
                   int it,
                   Projection projs[N_LAYER + N_DISK],
                   bool disk,
                   bool overlap)
    : settings_(settings) {
  seedIndex_ = iSeed;

  overlap_ = overlap;
  disk_ = disk;
  assert(!(disk && overlap));
  barrel_ = (!disk) && (!overlap);
  triplet_ = false;

  trackletIndex_ = -1;
  TCIndex_ = -1;

  assert(disk_ || barrel_ || overlap_);

  if (barrel_ && middleFPGAStub == nullptr)
    assert(innerFPGAStub->l1tstub()->layer() < N_LAYER);

  innerFPGAStub_ = innerFPGAStub;
  middleFPGAStub_ = middleFPGAStub;
  outerFPGAStub_ = outerFPGAStub;

  trackpars_.init(rinv, phi0, d0, t, z0);

  trackparsapprox_.init(rinvapprox, phi0approx, d0approx, tapprox, z0approx);

  fpgapars_.rinv().set(irinv, settings_.nbitsrinv(), false, __LINE__, __FILE__);
  fpgapars_.phi0().set(iphi0, settings_.nbitsphi0(), false, __LINE__, __FILE__);
  fpgapars_.d0().set(id0, settings_.nbitsd0(), false, __LINE__, __FILE__);
  fpgapars_.z0().set(iz0, settings_.nbitsz0(), false, __LINE__, __FILE__);
  fpgapars_.t().set(it, settings_.nbitst(), false, __LINE__, __FILE__);

  fpgatrack_ = nullptr;

  triplet_ = (seedIndex_ >= 8);

  //fill projection layers
  for (unsigned int i = 0; i < N_LAYER - 2; i++) {
    projlayer_[i] = settings.projlayers(seedIndex_, i);
  }

  //fill projection disks
  for (unsigned int i = 0; i < N_DISK; i++) {
    projdisk_[i] = settings.projdisks(seedIndex_, i);
  }

  //Handle projections to the layers
  for (unsigned int i = 0; i < N_LAYER - 2; i++) {
    if (projlayer_[i] == 0)
      continue;
    if (!projs[projlayer_[i] - 1].valid())
      continue;

    proj_[projlayer_[i] - 1] = projs[projlayer_[i] - 1];
  }
  //Now handle projections to the disks
  for (unsigned int i = 0; i < N_DISK; i++) {
    if (projdisk_[i] == 0)
      continue;
    if (!projs[N_LAYER + projdisk_[i] - 1].valid())
      continue;

    proj_[N_LAYER + projdisk_[i] - 1] = projs[N_LAYER + projdisk_[i] - 1];
  }

  ichisqrphifit_.set(-1, 8, false);
  ichisqrzfit_.set(-1, 8, false);
}

int Tracklet::tpseed() {
  set<int> tpset;

  set<int> tpsetstubinner;
  set<int> tpsetstubouter;

  vector<int> tps = innerFPGAStub_->l1tstub()->tps();
  for (auto tp : tps) {
    if (tp != 0) {
      tpsetstubinner.insert(tp);
      tpset.insert(abs(tp));
    }
  }

  tps = outerFPGAStub_->l1tstub()->tps();
  for (auto tp : tps) {
    if (tp != 0) {
      tpsetstubouter.insert(tp);
      tpset.insert(abs(tp));
    }
  }

  for (auto& tp : tpset) {
    if (tpsetstubinner.find(tp) != tpsetstubinner.end() && tpsetstubinner.find(-tp) != tpsetstubinner.end() &&
        tpsetstubouter.find(tp) != tpsetstubouter.end() && tpsetstubouter.find(-tp) != tpsetstubouter.end()) {
      return tp;
    }
  }
  return 0;
}

bool Tracklet::stubtruthmatch(const L1TStub* stub) {
  set<int> tpset;
  set<int> tpsetstub;
  set<int> tpsetstubinner;
  set<int> tpsetstubouter;

  vector<int> tps = stub->tps();
  for (auto tp : tps) {
    if (tp != 0) {
      tpsetstub.insert(tp);
      tpset.insert(abs(tp));
    }
  }
  tps = innerFPGAStub_->l1tstub()->tps();
  for (auto tp : tps) {
    if (tp != 0) {
      tpsetstubinner.insert(tp);
      tpset.insert(abs(tp));
    }
  }
  tps = outerFPGAStub_->l1tstub()->tps();
  for (auto tp : tps) {
    if (tp != 0) {
      tpsetstubouter.insert(tp);
      tpset.insert(abs(tp));
    }
  }

  for (auto tp : tpset) {
    if (tpsetstub.find(tp) != tpsetstub.end() && tpsetstub.find(-tp) != tpsetstub.end() &&
        tpsetstubinner.find(tp) != tpsetstubinner.end() && tpsetstubinner.find(-tp) != tpsetstubinner.end() &&
        tpsetstubouter.find(tp) != tpsetstubouter.end() && tpsetstubouter.find(-tp) != tpsetstubouter.end()) {
      return true;
    }
  }

  return false;
}

std::string Tracklet::addressstr() {
  std::string str;
  str = innerFPGAStub_->phiregionaddressstr() + "|";
  if (middleFPGAStub_) {
    str += middleFPGAStub_->phiregionaddressstr() + "|";
  }
  str += outerFPGAStub_->phiregionaddressstr();

  return str;
}

std::string Tracklet::trackletparstr() {
  if (settings_.writeoutReal()) {
    std::string oss = std::to_string(fpgapars_.rinv().value() * settings_.krinvpars()) + " " +
                      std::to_string(fpgapars_.phi0().value() * settings_.kphi0pars()) + " " +
                      std::to_string(fpgapars_.d0().value() * settings_.kd0pars()) + " " +
                      std::to_string(fpgapars_.z0().value() * settings_.kz()) + " " +
                      std::to_string(fpgapars_.t().value() * settings_.ktpars());
    return oss;
  } else {
    std::string str = innerFPGAStub_->stubindex().str() + "|";
    if (middleFPGAStub_) {
      str += middleFPGAStub_->stubindex().str() + "|";
    }
    str += outerFPGAStub_->stubindex().str() + "|" + fpgapars_.rinv().str() + "|" + fpgapars_.phi0().str() + "|";
    if (middleFPGAStub_)
      str += fpgapars_.d0().str() + "|";
    str += fpgapars_.z0().str() + "|" + fpgapars_.t().str();
    return str;
  }
}

std::string Tracklet::vmstrlayer(int layer, unsigned int allstubindex) {
  FPGAWord index;
  if (allstubindex >= (1 << 7)) {
    edm::LogPrint("Tracklet") << "Warning projection number too large!";
    index.set((1 << 7) - 1, 7, true, __LINE__, __FILE__);
  } else {
    index.set(allstubindex, 7, true, __LINE__, __FILE__);
  }

  // This is a shortcut.
  //int irinvvm=16+(fpgarinv().value()>>(fpgarinv().nbits()-5));
  // rinv is not directly available in the TrackletProjection.
  // can be inferred from phi derivative: rinv = - phider * 2
  int tmp_irinv = proj_[layer - 1].fpgaphiprojder().value() * (-2);
  int nbits_irinv = proj_[layer - 1].fpgaphiprojder().nbits() + 1;

  // irinv in VMProjection:
  // top 5 bits of rinv and shifted to be positive
  int irinvvm = 16 + (tmp_irinv >> (nbits_irinv - 5));

  if (settings_.extended() && (irinvvm > 31)) {  //TODO - displaced tracking should protect against this
    edm::LogPrint("Tracklet") << "Warning irinvvm too large:" << irinvvm;
    irinvvm = 31;
  }

  assert(irinvvm >= 0);
  assert(irinvvm < 32);
  FPGAWord tmp;
  tmp.set(irinvvm, 5, true, __LINE__, __FILE__);
  std::string oss = index.str() + "|" + proj_[layer - 1].fpgarzbin1projvm().str() + "|" +
                    proj_[layer - 1].fpgarzbin2projvm().str() + "|" + proj_[layer - 1].fpgafinerzvm().str() + "|" +
                    proj_[layer - 1].fpgafinephivm().str() + "|" + tmp.str() + "|" + std::to_string(PSseed());
  return oss;
}

std::string Tracklet::vmstrdisk(int disk, unsigned int allstubindex) {
  FPGAWord index;
  if (allstubindex >= (1 << 7)) {
    edm::LogPrint("Tracklet") << "Warning projection number too large!";
    index.set((1 << 7) - 1, 7, true, __LINE__, __FILE__);
  } else {
    index.set(allstubindex, 7, true, __LINE__, __FILE__);
  }
  std::string oss =
      index.str() + "|" + proj_[N_LAYER + disk - 1].fpgarzbin1projvm().str() + "|" +
      proj_[N_LAYER + disk - 1].fpgarzbin2projvm().str() + "|" + proj_[N_LAYER + disk - 1].fpgafinerzvm().str() + "|" +
      proj_[N_LAYER + disk - 1].fpgafinephivm().str() + "|" + proj_[N_LAYER + disk - 1].getBendIndex().str();
  return oss;
}

std::string Tracklet::trackletprojstr(int layer) const {
  assert(layer > 0 && layer <= N_LAYER);
  FPGAWord tmp;
  if (trackletIndex_ < 0 || trackletIndex_ > (int)settings_.ntrackletmax()) {
    throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " trackletIndex_ = " << trackletIndex_;
  }
  tmp.set(trackletIndex_, settings_.nbitstrackletindex(), true, __LINE__, __FILE__);
  FPGAWord tcid;
  tcid.set(TCIndex_, settings_.nbitstcindex(), true, __LINE__, __FILE__);

  std::string oss = tcid.str() + "|" + tmp.str() + "|" + proj_[layer - 1].fpgaphiproj().str() + "|" +
                    proj_[layer - 1].fpgarzproj().str() + "|" + proj_[layer - 1].fpgaphiprojder().str() + "|" +
                    proj_[layer - 1].fpgarzprojder().str();
  return oss;
}

std::string Tracklet::trackletprojstrD(int disk) const {
  assert(abs(disk) <= N_DISK);
  FPGAWord tmp;
  if (trackletIndex_ < 0 || trackletIndex_ > (int)settings_.ntrackletmax()) {
    throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " trackletIndex_ = " << trackletIndex_;
  }
  tmp.set(trackletIndex_, settings_.nbitstrackletindex(), true, __LINE__, __FILE__);
  FPGAWord tcid;
  if (settings_.extended()) {
    tcid.set(TCIndex_, 8, true, __LINE__, __FILE__);
  } else {
    tcid.set(TCIndex_, 7, true, __LINE__, __FILE__);
  }
  std::string oss = tcid.str() + "|" + tmp.str() + "|" + proj_[N_LAYER + abs(disk) - 1].fpgaphiproj().str() + "|" +
                    proj_[N_LAYER + abs(disk) - 1].fpgarzproj().str() + "|" +
                    proj_[N_LAYER + abs(disk) - 1].fpgaphiprojder().str() + "|" +
                    proj_[N_LAYER + abs(disk) - 1].fpgarzprojder().str();
  return oss;
}

void Tracklet::addMatch(unsigned int layerdisk,
                        int ideltaphi,
                        int ideltarz,
                        double dphi,
                        double drz,
                        double dphiapprox,
                        double drzapprox,
                        int stubid,
                        const trklet::Stub* stubptr) {
  assert(layerdisk < N_LAYER + N_DISK);
  resid_[layerdisk].init(settings_, layerdisk, ideltaphi, ideltarz, stubid, dphi, drz, dphiapprox, drzapprox, stubptr);
}

std::string Tracklet::fullmatchstr(int layer) {
  assert(layer > 0 && layer <= N_LAYER);

  FPGAWord tmp;
  if (trackletIndex_ < 0 || trackletIndex_ > (int)settings_.ntrackletmax()) {
    throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " trackletIndex_ = " << trackletIndex_;
  }
  tmp.set(trackletIndex_, settings_.nbitstrackletindex(), true, __LINE__, __FILE__);
  FPGAWord tcid;
  tcid.set(TCIndex_, settings_.nbitstcindex(), true, __LINE__, __FILE__);
  std::string oss = tcid.str() + "|" + tmp.str() + "|" + resid_[layer - 1].fpgastubid().str() + "|" +
                    resid_[layer - 1].stubptr()->r().str() + "|" + resid_[layer - 1].fpgaphiresid().str() + "|" +
                    resid_[layer - 1].fpgarzresid().str();
  return oss;
}

std::string Tracklet::fullmatchdiskstr(int disk) {
  assert(disk > 0 && disk <= N_DISK);

  FPGAWord tmp;
  if (trackletIndex_ < 0 || trackletIndex_ > (int)settings_.ntrackletmax()) {
    throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " trackletIndex_ = " << trackletIndex_;
  }
  tmp.set(trackletIndex_, settings_.nbitstrackletindex(), true, __LINE__, __FILE__);
  FPGAWord tcid;
  tcid.set(TCIndex_, settings_.nbitstcindex(), true, __LINE__, __FILE__);
  const FPGAWord& stubr = resid_[N_LAYER + disk - 1].stubptr()->r();
  const bool isPS = resid_[N_LAYER + disk - 1].stubptr()->isPSmodule();
  std::string oss = tcid.str() + "|" + tmp.str() + "|" + resid_[N_LAYER + disk - 1].fpgastubid().str() + "|" +
                    (isPS ? stubr.str() : ("00000000" + stubr.str())) + "|" +
                    resid_[N_LAYER + disk - 1].fpgaphiresid().str() + "|" +
                    resid_[N_LAYER + disk - 1].fpgarzresid().str();
  return oss;
}

std::vector<const L1TStub*> Tracklet::getL1Stubs() {
  std::vector<const L1TStub*> tmp;

  if (innerFPGAStub_)
    tmp.push_back(innerFPGAStub_->l1tstub());
  if (middleFPGAStub_)
    tmp.push_back(middleFPGAStub_->l1tstub());
  if (outerFPGAStub_)
    tmp.push_back(outerFPGAStub_->l1tstub());

  for (const auto& iresid : resid_) {
    if (iresid.valid())
      tmp.push_back(iresid.stubptr()->l1tstub());
  }

  return tmp;
}

std::map<int, int> Tracklet::getStubIDs() {
  std::map<int, int> stubIDs;

  // For future reference, *resid_[i] uses i as the absolute stub index. (0-5 for barrel, 0-4 for disk)
  // On the other hand, proj*_[i] uses i almost like *resid_[i], except the *seeding* layer indices are removed entirely.
  // E.g. An L3L4 track has 0=L1, 1=L2, 2=L4, 3=L5 for the barrels (for proj*_[i])

  if (innerFPGAStub_)
    assert(innerFPGAStub_->stubindex().nbits() == N_BITSMEMADDRESS);
  if (middleFPGAStub_)
    assert(middleFPGAStub_->stubindex().nbits() == N_BITSMEMADDRESS);
  if (outerFPGAStub_)
    assert(outerFPGAStub_->stubindex().nbits() == N_BITSMEMADDRESS);

  if (barrel_) {
    for (int i = 0; i < N_LAYER; i++) {
      //check barrel
      if (resid_[i].valid()) {
        // two extra bits to indicate if the matched stub is local or from neighbor
        int location = 1;  // local
        location <<= resid_[i].fpgastubid().nbits();

        stubIDs[1 + i] = resid_[i].fpgastubid().value() + location;
      }

      //check disk
      if (i >= N_DISK)
        continue;  //i=[0..4] for disks
      if (resid_[N_LAYER + i].valid()) {
        if (i == 3 && resid_[0].valid() && innerFPGAStub_->layer().value() == 1)
          continue;  // Don't add D4 if track has L1 stub
        // two extra bits to indicate if the matched stub is local or from neighbor
        int location = 1;  // local
        location <<= resid_[N_LAYER + i].fpgastubid().nbits();

        if (itfit().value() < 0) {
          stubIDs[-(N_LAYER + N_DISK) - i] = resid_[N_LAYER + i].fpgastubid().value() + location;
        } else {
          stubIDs[N_LAYER + N_DISK + i] = resid_[N_LAYER + i].fpgastubid().value() + location;
        }
      }
    }

    //get stubs making up tracklet
    if (innerFPGAStub_)
      stubIDs[innerFPGAStub_->layer().value() + 1] = innerFPGAStub_->phiregionaddress() + (1 << 10);
    if (middleFPGAStub_)
      stubIDs[middleFPGAStub_->layer().value() + 1] = middleFPGAStub_->phiregionaddress() + (1 << 10);
    if (outerFPGAStub_)
      stubIDs[outerFPGAStub_->layer().value() + 1] = outerFPGAStub_->phiregionaddress() + (1 << 10);

  } else if (disk_) {
    for (int i = 0; i < N_DISK; i++) {
      //check barrel
      if (resid_[i].valid()) {
        // two extra bits to indicate if the matched stub is local or from neighbor
        int location = 1;  // local
        location <<= resid_[i].fpgastubid().nbits();

        stubIDs[1 + i] = resid_[i].fpgastubid().value() + location;
      }

      //check disks
      if (i == 4 && resid_[1].valid())
        continue;  // Don't add D5 if track has L2 stub
      if (resid_[N_LAYER + i].valid()) {
        // two extra bits to indicate if the matched stub is local or from neighbor
        int location = 1;  // local
        location <<= resid_[N_LAYER + i].fpgastubid().nbits();

        if (innerFPGAStub_->l1tstub()->disk() < 0) {
          stubIDs[-11 - i] = resid_[N_LAYER + i].fpgastubid().value() + location;
        } else {
          stubIDs[11 + i] = resid_[N_LAYER + i].fpgastubid().value() + location;
        }
      }
    }

    //get stubs making up tracklet
    if (innerFPGAStub_->disk().value() < 0) {  //negative side runs 6-10
      if (innerFPGAStub_)
        stubIDs[innerFPGAStub_->disk().value() - 10] = innerFPGAStub_->phiregionaddress() + (1 << 10);
      if (middleFPGAStub_)
        stubIDs[middleFPGAStub_->disk().value() - 10] = middleFPGAStub_->phiregionaddress() + (1 << 10);
      if (outerFPGAStub_)
        stubIDs[outerFPGAStub_->disk().value() - 10] = outerFPGAStub_->phiregionaddress() + (1 << 10);
    } else {  // positive side runs 11-15]
      if (innerFPGAStub_)
        stubIDs[innerFPGAStub_->disk().value() + 10] = innerFPGAStub_->phiregionaddress() + (1 << 10);
      if (middleFPGAStub_)
        stubIDs[middleFPGAStub_->disk().value() + 10] = middleFPGAStub_->phiregionaddress() + (1 << 10);
      if (outerFPGAStub_)
        stubIDs[outerFPGAStub_->disk().value() + 10] = outerFPGAStub_->phiregionaddress() + (1 << 10);
    }

  } else if (overlap_) {
    for (int i = 0; i < N_DISK; i++) {
      //check barrel
      if (resid_[i].valid()) {
        // two extra bits to indicate if the matched stub is local or from neighbor
        int location = 1;  // local
        location <<= resid_[i].fpgastubid().nbits();

        stubIDs[1 + i] = resid_[i].fpgastubid().value() + location;
      }

      //check disks
      if (resid_[N_LAYER + i].valid()) {
        // two extra bits to indicate if the matched stub is local or from neighbor
        int location = 1;  // local
        location <<= resid_[N_LAYER + i].fpgastubid().nbits();

        if (innerFPGAStub_->l1tstub()->disk() < 0) {  // if negative overlap
          if (innerFPGAStub_->layer().value() != 2 || !resid_[0].valid() ||
              i != 3) {  // Don't add D4 if this is an L3L2 track with an L1 stub
            stubIDs[-11 - i] = resid_[N_LAYER + i].fpgastubid().value() + location;
          }
        } else {
          if (innerFPGAStub_->layer().value() != 2 || !resid_[0].valid() || i != 3) {
            stubIDs[11 + i] = resid_[N_LAYER + i].fpgastubid().value() + location;
          }
        }
      }
    }

    //get stubs making up tracklet

    if (innerFPGAStub_->layer().value() == 2) {  // L3L2 track
      if (innerFPGAStub_)
        stubIDs[innerFPGAStub_->layer().value() + 1] = innerFPGAStub_->phiregionaddress() + (1 << 10);
      if (middleFPGAStub_)
        stubIDs[middleFPGAStub_->layer().value() + 1] = middleFPGAStub_->phiregionaddress() + (1 << 10);
      if (outerFPGAStub_)
        stubIDs[outerFPGAStub_->layer().value() + 1] = outerFPGAStub_->phiregionaddress() + (1 << 10);
    } else if (innerFPGAStub_->disk().value() < 0) {  //negative side runs -11 - -15
      if (innerFPGAStub_)
        stubIDs[innerFPGAStub_->disk().value() - 10] = innerFPGAStub_->phiregionaddress() + (1 << 10);
      if (middleFPGAStub_)
        stubIDs[middleFPGAStub_->layer().value() + 1] = middleFPGAStub_->phiregionaddress() + (1 << 10);
      if (outerFPGAStub_)
        stubIDs[outerFPGAStub_->layer().value() + 1] = outerFPGAStub_->phiregionaddress() + (1 << 10);
    } else {  // positive side runs 11-15]
      if (innerFPGAStub_)
        stubIDs[innerFPGAStub_->disk().value() + 10] = innerFPGAStub_->phiregionaddress() + (1 << 10);
      if (middleFPGAStub_)
        stubIDs[middleFPGAStub_->layer().value() + 1] = middleFPGAStub_->phiregionaddress() + (1 << 10);
      if (outerFPGAStub_)
        stubIDs[outerFPGAStub_->layer().value() + 1] = outerFPGAStub_->phiregionaddress() + (1 << 10);
    }
  }

  return stubIDs;
}

void Tracklet::setFitPars(double rinvfit,
                          double phi0fit,
                          double d0fit,
                          double tfit,
                          double z0fit,
                          double chisqrphifit,
                          double chisqrzfit,
                          double rinvfitexact,
                          double phi0fitexact,
                          double d0fitexact,
                          double tfitexact,
                          double z0fitexact,
                          double chisqrphifitexact,
                          double chisqrzfitexact,
                          int irinvfit,
                          int iphi0fit,
                          int id0fit,
                          int itfit,
                          int iz0fit,
                          int ichisqrphifit,
                          int ichisqrzfit,
                          int hitpattern,
                          const vector<const L1TStub*>& l1stubs) {
  fitpars_.init(rinvfit, phi0fit, d0fit, tfit, z0fit);
  chisqrphifit_ = chisqrphifit;
  chisqrzfit_ = chisqrzfit;

  fitparsexact_.init(rinvfitexact, phi0fitexact, d0fitexact, tfitexact, z0fitexact);
  chisqrphifitexact_ = chisqrphifitexact;
  chisqrzfitexact_ = chisqrzfitexact;

  if (irinvfit > (1 << 14))
    irinvfit = (1 << 14);
  if (irinvfit <= -(1 << 14))
    irinvfit = -(1 << 14) + 1;
  fpgafitpars_.rinv().set(irinvfit, 15, false, __LINE__, __FILE__);
  fpgafitpars_.phi0().set(iphi0fit, 19, false, __LINE__, __FILE__);
  fpgafitpars_.d0().set(id0fit, 19, false, __LINE__, __FILE__);
  fpgafitpars_.t().set(itfit, 14, false, __LINE__, __FILE__);

  if (iz0fit >= (1 << (settings_.nbitsz0() - 1))) {
    iz0fit = (1 << (settings_.nbitsz0() - 1)) - 1;
  }

  if (iz0fit <= -(1 << (settings_.nbitsz0() - 1))) {
    iz0fit = 1 - (1 << (settings_.nbitsz0() - 1));
  }

  fpgafitpars_.z0().set(iz0fit, settings_.nbitsz0(), false, __LINE__, __FILE__);
  ichisqrphifit_.set(ichisqrphifit, 8, true, __LINE__, __FILE__);
  ichisqrzfit_.set(ichisqrzfit, 8, true, __LINE__, __FILE__);

  hitpattern_ = hitpattern;

  fpgatrack_ = std::make_unique<Track>(makeTrack(l1stubs));
}

const std::string Tracklet::layerstubstr(const unsigned layer) const {
  assert(layer < N_LAYER);

  std::stringstream oss("");
  if (!resid_[layer].valid())
    oss << "0|0000000|0000000000|0000000|000000000000|000000000";
  else {
    if (trackIndex_ < 0 || trackIndex_ > (int)settings_.ntrackletmax()) {
      cout << "trackIndex_ = " << trackIndex_ << endl;
      assert(0);
    }
    const FPGAWord tmp(trackIndex_, settings_.nbitstrackletindex(), true, __LINE__, __FILE__);
    oss << "1|";  // valid bit
    oss << tmp.str() << "|";
    oss << resid_[layer].fpgastubid().str() << "|";
    oss << resid_[layer].stubptr()->r().str() << "|";
    oss << resid_[layer].fpgaphiresid().str() << "|";
    oss << resid_[layer].fpgarzresid().str();
  }

  return oss.str();
}

const std::string Tracklet::diskstubstr(const unsigned disk) const {
  assert(disk < N_DISK);

  std::stringstream oss("");
  if (!resid_[N_LAYER + disk].valid())
    oss << "0|0000000|0000000000|000000000000|000000000000|0000000";
  else {
    if (trackIndex_ < 0 || trackIndex_ > (int)settings_.ntrackletmax()) {
      cout << "trackIndex_ = " << trackIndex_ << endl;
      assert(0);
    }
    const FPGAWord tmp(trackIndex_, settings_.nbitstrackletindex(), true, __LINE__, __FILE__);
    const FPGAWord& stubr = resid_[N_LAYER + disk].stubptr()->r();
    const bool isPS = resid_[N_LAYER + disk].stubptr()->isPSmodule();
    oss << "1|";  // valid bit
    oss << tmp.str() << "|";
    oss << resid_[N_LAYER + disk].fpgastubid().str() << "|";
    oss << (isPS ? stubr.str() : ("00000000" + stubr.str())) << "|";
    oss << resid_[N_LAYER + disk].fpgaphiresid().str() << "|";
    oss << resid_[N_LAYER + disk].fpgarzresid().str();
  }

  return oss.str();
}

std::string Tracklet::trackfitstr() const {
  const unsigned maxNHits = 8;
  const unsigned nBitsPerHit = 3;
  vector<string> stub(maxNHits, "0");
  string hitmap(maxNHits * nBitsPerHit, '0');

  // Assign stub strings for each of the possible projections for each seed.
  // The specific layers/disks for a given seed are determined by the wiring.
  switch (seedIndex()) {
    case 0:                       // L1L2
      stub[0] = layerstubstr(2);  // L3
      stub[1] = layerstubstr(3);  // L4
      stub[2] = layerstubstr(4);  // L5
      stub[3] = layerstubstr(5);  // L6

      stub[4] = diskstubstr(0);  // D1
      stub[5] = diskstubstr(1);  // D2
      stub[6] = diskstubstr(2);  // D3
      stub[7] = diskstubstr(3);  // D4

      break;

    case 1:                       // L2L3
      stub[0] = layerstubstr(0);  // L1
      stub[1] = layerstubstr(3);  // L4
      stub[2] = layerstubstr(4);  // L5

      stub[3] = diskstubstr(0);  // D1
      stub[4] = diskstubstr(1);  // D2
      stub[5] = diskstubstr(2);  // D3
      stub[6] = diskstubstr(3);  // D4

      break;

    case 2:                       // L3L4
      stub[0] = layerstubstr(0);  // L1
      stub[1] = layerstubstr(1);  // L2
      stub[2] = layerstubstr(4);  // L5
      stub[3] = layerstubstr(5);  // L6

      stub[4] = diskstubstr(0);  // D1
      stub[5] = diskstubstr(1);  // D2

      break;

    case 3:                       // L5L6
      stub[0] = layerstubstr(0);  // L1
      stub[1] = layerstubstr(1);  // L2
      stub[2] = layerstubstr(2);  // L3
      stub[3] = layerstubstr(3);  // L4

      break;

    case 4:                       // D1D2
      stub[0] = layerstubstr(0);  // L1
      stub[1] = layerstubstr(1);  // L2

      stub[2] = diskstubstr(2);  // D3
      stub[3] = diskstubstr(3);  // D4
      stub[4] = diskstubstr(4);  // D5

      break;

    case 5:                       // D3D4
      stub[0] = layerstubstr(0);  // L1

      stub[1] = diskstubstr(0);  // D1
      stub[2] = diskstubstr(1);  // D2
      stub[3] = diskstubstr(4);  // D5

      break;

    case 6:                      // L1D1
      stub[0] = diskstubstr(1);  // D2
      stub[1] = diskstubstr(2);  // D3
      stub[2] = diskstubstr(3);  // D4
      stub[3] = diskstubstr(4);  // D5

      break;

    case 7:                       // L2D1
      stub[0] = layerstubstr(0);  // L1

      stub[1] = diskstubstr(1);  // D2
      stub[2] = diskstubstr(2);  // D3
      stub[3] = diskstubstr(3);  // D4

      break;
  }

  // Only one hit per layer/disk is allowed currently, so the hit map for a
  // given layer/disk is just equal to the valid bit of the corresponding stub
  // string, which is the first character.
  for (unsigned i = 0; i < maxNHits; i++)
    hitmap[i * nBitsPerHit + 2] = stub[i][0];

  std::string oss("");
  //Binary print out
  if (!settings_.writeoutReal()) {
    const FPGAWord tmp(getISeed(), settings_.nbitsseed(), true, __LINE__, __FILE__);

    oss += "1|";  // valid bit
    oss += tmp.str() + "|";
    oss += innerFPGAStub()->stubindex().str() + "|";
    oss += outerFPGAStub()->stubindex().str() + "|";
    oss += fpgapars_.rinv().str() + "|";
    oss += fpgapars_.phi0().str() + "|";
    oss += fpgapars_.z0().str() + "|";
    oss += fpgapars_.t().str() + "|";
    oss += hitmap;
    for (unsigned i = 0; i < maxNHits; i++)
      // If a valid stub string was never assigned, then that stub is not
      // included in the output.
      if (stub[i] != "0")
        oss += "|" + stub[i];
  }

  return oss;
}

// Create a Track object from stubs & digitized track helix params

Track Tracklet::makeTrack(const vector<const L1TStub*>& l1stubs) {
  assert(fit());

  // Digitized track helix params
  TrackPars<int> ipars(fpgafitpars_.rinv().value(),
                       fpgafitpars_.phi0().value(),
                       fpgafitpars_.d0().value(),
                       fpgafitpars_.t().value(),
                       fpgafitpars_.z0().value());

  // If fitter produced no stub list, take it from original tracklet.
  vector<const L1TStub*> tmp = l1stubs.empty() ? getL1Stubs() : l1stubs;

  vector<L1TStub> tmp2;

  tmp2.reserve(tmp.size());
  for (auto stub : tmp) {
    tmp2.push_back(*stub);
  }

  Track tmpTrack(ipars,
                 ichisqrphifit_.value(),
                 ichisqrzfit_.value(),
                 chisqrphifit_,
                 chisqrzfit_,
                 hitpattern_,
                 getStubIDs(),
                 tmp2,
                 getISeed());

  return tmpTrack;
}

int Tracklet::layer() const {
  int l1 = (innerFPGAStub_ && innerFPGAStub_->layerdisk() < N_LAYER) ? innerFPGAStub_->l1tstub()->layerdisk() + 1 : 999,
      l2 = (middleFPGAStub_ && middleFPGAStub_->layerdisk() < N_LAYER) ? middleFPGAStub_->l1tstub()->layerdisk() + 1
                                                                       : 999,
      l3 = (outerFPGAStub_ && outerFPGAStub_->layerdisk() < N_LAYER) ? outerFPGAStub_->l1tstub()->layerdisk() + 1 : 999,
      l = min(min(l1, l2), l3);
  return (l < 999 ? l : 0);
}

int Tracklet::disk() const {
  int d1 = (innerFPGAStub_ && (innerFPGAStub_->layerdisk() >= N_LAYER)) ? innerFPGAStub_->l1tstub()->disk() : 999,
      d2 = (middleFPGAStub_ && (middleFPGAStub_->layerdisk() >= N_LAYER)) ? middleFPGAStub_->l1tstub()->disk() : 999,
      d3 = (outerFPGAStub_ && (outerFPGAStub_->layerdisk() >= N_LAYER)) ? outerFPGAStub_->l1tstub()->disk() : 999,
      d = 999;
  if (abs(d1) < min(abs(d2), abs(d3)))
    d = d1;
  if (abs(d2) < min(abs(d1), abs(d3)))
    d = d2;
  if (abs(d3) < min(abs(d1), abs(d2)))
    d = d3;
  return (d < 999 ? d : 0);
}

void Tracklet::setTrackletIndex(unsigned int index) {
  trackletIndex_ = index;
  assert(index <= settings_.ntrackletmax());
}

int Tracklet::getISeed() const {
  const int iSeed = TCIndex_ >> settings_.nbitsitc();
  assert(iSeed >= 0 && iSeed <= (int)N_SEED);
  return iSeed;
}

int Tracklet::getITC() const {
  const int iSeed = getISeed(), iTC = TCIndex_ - (iSeed << settings_.nbitsitc());
  assert(iTC >= 0 && iTC <= 14);
  return iTC;
}

void Tracklet::setTrackIndex(int index) {
  trackIndex_ = index;
  assert(index <= (int)settings_.ntrackletmax());
}

int Tracklet::trackIndex() const { return trackIndex_; }
