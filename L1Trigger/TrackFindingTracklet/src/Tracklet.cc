#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include "L1Trigger/TrackFindingTracklet/interface/Track.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <sstream>

using namespace std;
using namespace trklet;

Tracklet::Tracklet(Settings const& settings,
                   const L1TStub* innerStub,
                   const L1TStub* middleStub,
                   const L1TStub* outerStub,
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
                   LayerProjection layerprojs[N_PROJ],
                   DiskProjection diskprojs[N_PROJ],
                   bool disk,
                   bool overlap)
    : settings_(settings) {
  overlap_ = overlap;
  disk_ = disk;
  assert(!(disk && overlap));
  barrel_ = (!disk) && (!overlap);
  triplet_ = false;

  trackletIndex_ = -1;
  TCIndex_ = -1;

  assert(disk_ || barrel_ || overlap_);

  if (barrel_ && middleStub == nullptr)
    assert(innerStub->layer() < N_LAYER);

  innerStub_ = innerStub;
  middleStub_ = middleStub;
  outerStub_ = outerStub;
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

  if (innerStub_)
    assert(innerStub_->layer() < N_LAYER || innerStub_->disk() < N_DISK);
  if (middleStub_)
    assert(middleStub_->layer() < N_LAYER || middleStub_->disk() < N_DISK);
  if (outerStub_)
    assert(outerStub_->layer() < N_LAYER || outerStub_->disk() < N_DISK);

  seedIndex_ = calcSeedIndex();

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
    if (!layerprojs[i].valid())
      continue;

    layerproj_[projlayer_[i] - 1] = layerprojs[i];
  }
  //Now handle projections to the disks
  for (unsigned int i = 0; i < N_DISK; i++) {
    if (projdisk_[i] == 0)
      continue;
    if (!diskprojs[i].valid())
      continue;

    diskproj_[projdisk_[i] - 1] = diskprojs[i];
  }

  ichisqrphifit_.set(-1, 8, false);
  ichisqrzfit_.set(-1, 8, false);
}

int Tracklet::tpseed() {
  set<int> tpset;

  set<int> tpsetstubinner;
  set<int> tpsetstubouter;

  vector<int> tps = innerStub_->tps();
  for (auto tp : tps) {
    if (tp != 0) {
      tpsetstubinner.insert(tp);
      tpset.insert(abs(tp));
    }
  }

  tps = outerStub_->tps();
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
  tps = innerStub_->tps();
  for (auto tp : tps) {
    if (tp != 0) {
      tpsetstubinner.insert(tp);
      tpset.insert(abs(tp));
    }
  }
  tps = outerStub_->tps();
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
    if (middleFPGAStub_) str += fpgapars_.d0().str() + "|";
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
  int tmp_irinv = layerproj_[layer - 1].fpgaphiprojder().value() * (-2);
  int nbits_irinv = layerproj_[layer - 1].fpgaphiprojder().nbits() + 1;

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
  std::string oss = index.str() + "|" + layerproj_[layer - 1].fpgazbin1projvm().str() + "|" +
                    layerproj_[layer - 1].fpgazbin2projvm().str() + "|" + layerproj_[layer - 1].fpgafinezvm().str() +
                    "|" + tmp.str() + "|" + std::to_string(PSseed());
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
  std::string oss = index.str() + "|" + diskproj_[disk - 1].fpgarbin1projvm().str() + "|" +
                    diskproj_[disk - 1].fpgarbin2projvm().str() + "|" + diskproj_[disk - 1].fpgafinervm().str() + "|" +
                    diskproj_[disk - 1].getBendIndex().str();
  return oss;
}

std::string Tracklet::trackletprojstr(int layer) const {
  assert(layer > 0 && layer <= N_LAYER);
  FPGAWord tmp;
  if (trackletIndex_ < 0 || trackletIndex_ > (int)settings_.ntrackletmax()) {
    throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " trackletIndex_ = " << trackletIndex_;
  }
  tmp.set(trackletIndex_, 7, true, __LINE__, __FILE__);
  FPGAWord tcid;
  if (settings_.extended()) {
    tcid.set(TCIndex_, 8, true, __LINE__, __FILE__);
  } else {
    tcid.set(TCIndex_, 7, true, __LINE__, __FILE__);
  }

  std::string oss = tcid.str() + "|" + tmp.str() + "|" + layerproj_[layer - 1].fpgaphiproj().str() + "|" +
                    layerproj_[layer - 1].fpgazproj().str() + "|" + layerproj_[layer - 1].fpgaphiprojder().str() + "|" +
                    layerproj_[layer - 1].fpgazprojder().str();
  return oss;
}

std::string Tracklet::trackletprojstrD(int disk) const {
  assert(abs(disk) <= N_DISK);
  FPGAWord tmp;
  if (trackletIndex_ < 0 || trackletIndex_ > (int)settings_.ntrackletmax()) {
    throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " trackletIndex_ = " << trackletIndex_;
  }
  tmp.set(trackletIndex_, 7, true, __LINE__, __FILE__);
  FPGAWord tcid;
  if (settings_.extended()) {
    tcid.set(TCIndex_, 8, true, __LINE__, __FILE__);
  } else {
    tcid.set(TCIndex_, 7, true, __LINE__, __FILE__);
  }
  std::string oss = tcid.str() + "|" + tmp.str() + "|" + diskproj_[abs(disk) - 1].fpgaphiproj().str() + "|" +
                    diskproj_[abs(disk) - 1].fpgarproj().str() + "|" + diskproj_[abs(disk) - 1].fpgaphiprojder().str() +
                    "|" + diskproj_[abs(disk) - 1].fpgarprojder().str();
  return oss;
}

void Tracklet::addMatch(int layer,
                        int ideltaphi,
                        int ideltaz,
                        double dphi,
                        double dz,
                        double dphiapprox,
                        double dzapprox,
                        int stubid,
                        double rstub,
                        const trklet::Stub* stubptr) {
  assert(layer > 0 && layer <= N_LAYER);
  layerresid_[layer - 1].init(
      settings_, layer, ideltaphi, ideltaz, stubid, dphi, dz, dphiapprox, dzapprox, rstub, stubptr);
}

void Tracklet::addMatchDisk(int disk,
                            int ideltaphi,
                            int ideltar,
                            double dphi,
                            double dr,
                            double dphiapprox,
                            double drapprox,
                            double alpha,
                            int stubid,
                            double zstub,
                            const trklet::Stub* stubptr) {
  assert(abs(disk) <= N_DISK);
  diskresid_[abs(disk) - 1].init(settings_,
                                 disk,
                                 ideltaphi,
                                 ideltar,
                                 stubid,
                                 dphi,
                                 dr,
                                 dphiapprox,
                                 drapprox,
                                 zstub,
                                 alpha,
                                 stubptr->alphanew(),
                                 stubptr);
}

int Tracklet::nMatches() {
  int nmatches = 0;

  for (const auto& ilayerresid : layerresid_) {
    if (ilayerresid.valid()) {
      nmatches++;
    }
  }

  return nmatches;
}

int Tracklet::nMatchesDisk() {
  int nmatches = 0;

  for (const auto& idiskresid : diskresid_) {
    if (idiskresid.valid()) {
      nmatches++;
    }
  }
  return nmatches;
}

std::string Tracklet::fullmatchstr(int layer) {
  assert(layer > 0 && layer <= N_LAYER);

  FPGAWord tmp;
  if (trackletIndex_ < 0 || trackletIndex_ > (int)settings_.ntrackletmax()) {
    throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " trackletIndex_ = " << trackletIndex_;
  }
  tmp.set(trackletIndex_, 7, true, __LINE__, __FILE__);
  FPGAWord tcid;
  if (settings_.extended()) {
    tcid.set(TCIndex_, 8, true, __LINE__, __FILE__);
  } else {
    tcid.set(TCIndex_, 7, true, __LINE__, __FILE__);
  }
  std::string oss = tcid.str() + "|" + tmp.str() + "|" + layerresid_[layer - 1].fpgastubid().str() + "|" +
                    layerresid_[layer - 1].fpgaphiresid().str() + "|" + layerresid_[layer - 1].fpgazresid().str();
  return oss;
}

std::string Tracklet::fullmatchdiskstr(int disk) {
  assert(disk > 0 && disk <= N_DISK);

  FPGAWord tmp;
  if (trackletIndex_ < 0 || trackletIndex_ > (int)settings_.ntrackletmax()) {
    throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " trackletIndex_ = " << trackletIndex_;
  }
  tmp.set(trackletIndex_, 7, true, __LINE__, __FILE__);
  FPGAWord tcid;
  if (settings_.extended()) {
    tcid.set(TCIndex_, 8, true, __LINE__, __FILE__);
  } else {
    tcid.set(TCIndex_, 7, true, __LINE__, __FILE__);
  }
  std::string oss = tcid.str() + "|" + tmp.str() + "|" + diskresid_[disk - 1].fpgastubid().str() + "|" +
                    diskresid_[disk - 1].fpgaphiresid().str() + "|" + diskresid_[disk - 1].fpgarresid().str();
  return oss;
}

std::vector<const L1TStub*> Tracklet::getL1Stubs() {
  std::vector<const L1TStub*> tmp;

  if (innerStub_)
    tmp.push_back(innerStub_);
  if (middleStub_)
    tmp.push_back(middleStub_);
  if (outerStub_)
    tmp.push_back(outerStub_);

  for (const auto& ilayerresid : layerresid_) {
    if (ilayerresid.valid())
      tmp.push_back(ilayerresid.stubptr()->l1tstub());
  }

  for (const auto& idiskresid : diskresid_) {
    if (idiskresid.valid())
      tmp.push_back(idiskresid.stubptr()->l1tstub());
  }

  return tmp;
}

std::map<int, int> Tracklet::getStubIDs() {
  std::map<int, int> stubIDs;

  // For future reference, *resid_[i] uses i as the absolute stub index. (0-5 for barrel, 0-4 for disk)
  // On the other hand, proj*_[i] uses i almost like *resid_[i], except the *seeding* layer indices are removed entirely.
  // E.g. An L3L4 track has 0=L1, 1=L2, 2=L4, 3=L5 for the barrels (for proj*_[i])

  if (innerFPGAStub_)
    assert(innerFPGAStub_->stubindex().nbits() == 7);
  if (middleFPGAStub_)
    assert(middleFPGAStub_->stubindex().nbits() == 7);
  if (outerFPGAStub_)
    assert(outerFPGAStub_->stubindex().nbits() == 7);

  if (barrel_) {
    for (int i = 0; i < N_LAYER; i++) {
      //check barrel
      if (layerresid_[i].valid()) {
        // two extra bits to indicate if the matched stub is local or from neighbor
        int location = 1;  // local
        location <<= layerresid_[i].fpgastubid().nbits();

        stubIDs[1 + i] = layerresid_[i].fpgastubid().value() + location;
      }

      //check disk
      if (i >= N_DISK)
        continue;  //i=[0..4] for disks
      if (diskresid_[i].valid()) {
        if (i == 3 && layerresid_[0].valid() && innerFPGAStub_->layer().value() == 1)
          continue;  // Don't add D4 if track has L1 stub
        // two extra bits to indicate if the matched stub is local or from neighbor
        int location = 1;  // local
        location <<= diskresid_[i].fpgastubid().nbits();

        if (itfit().value() < 0) {
          stubIDs[-11 - i] = diskresid_[i].fpgastubid().value() + location;
        } else {
          stubIDs[11 + i] = diskresid_[i].fpgastubid().value() + location;
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
      if (layerresid_[i].valid()) {
        // two extra bits to indicate if the matched stub is local or from neighbor
        int location = 1;  // local
        location <<= layerresid_[i].fpgastubid().nbits();

        stubIDs[1 + i] = layerresid_[i].fpgastubid().value() + location;
      }

      //check disks
      if (i == 4 && layerresid_[1].valid())
        continue;  // Don't add D5 if track has L2 stub
      if (diskresid_[i].valid()) {
        // two extra bits to indicate if the matched stub is local or from neighbor
        int location = 1;  // local
        location <<= diskresid_[i].fpgastubid().nbits();

        if (innerStub_->disk() < 0) {
          stubIDs[-11 - i] = diskresid_[i].fpgastubid().value() + location;
        } else {
          stubIDs[11 + i] = diskresid_[i].fpgastubid().value() + location;
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
      if (layerresid_[i].valid()) {
        // two extra bits to indicate if the matched stub is local or from neighbor
        int location = 1;  // local
        location <<= layerresid_[i].fpgastubid().nbits();

        stubIDs[1 + i] = layerresid_[i].fpgastubid().value() + location;
      }

      //check disks
      if (diskresid_[i].valid()) {
        // two extra bits to indicate if the matched stub is local or from neighbor
        int location = 1;  // local
        location <<= diskresid_[i].fpgastubid().nbits();

        if (innerStub_->disk() < 0) {  // if negative overlap
          if (innerFPGAStub_->layer().value() != 2 || !layerresid_[0].valid() ||
              i != 3) {  // Don't add D4 if this is an L3L2 track with an L1 stub
            stubIDs[-11 - i] = diskresid_[i].fpgastubid().value() + location;
          }
        } else {
          if (innerFPGAStub_->layer().value() != 2 || !layerresid_[0].valid() || i != 3) {
            stubIDs[11 + i] = diskresid_[i].fpgastubid().value() + location;
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

  fpgatrack_.reset(new Track(makeTrack(l1stubs)));
}

std::string Tracklet::trackfitstr() {
  string stubid0 = "111111111";
  string stubid1 = "111111111";
  string stubid2 = "111111111";
  string stubid3 = "111111111";

  if (isBarrel()) {
    if (layer() == 1) {
      if (layerresid_[2].valid()) {
        stubid0 = layerresid_[2].fpgastubid().str();
      }
      if (layerresid_[3].valid()) {
        stubid1 = layerresid_[3].fpgastubid().str();
      }
      if (layerresid_[4].valid()) {
        stubid2 = layerresid_[4].fpgastubid().str();
      }
      if (layerresid_[5].valid()) {
        stubid3 = layerresid_[5].fpgastubid().str();
      }
      if (diskresid_[0].valid()) {
        stubid3 = diskresid_[0].fpgastubid().str();
      }
      if (diskresid_[1].valid()) {
        stubid2 = diskresid_[1].fpgastubid().str();
      }
      if (diskresid_[2].valid()) {
        stubid1 = diskresid_[2].fpgastubid().str();
      }
      if (diskresid_[3].valid()) {
        stubid0 = diskresid_[3].fpgastubid().str();
      }
    }

    if (layer() == 3) {
      if (layerresid_[0].valid()) {
        stubid0 = layerresid_[0].fpgastubid().str();
      }
      if (layerresid_[1].valid()) {
        stubid1 = layerresid_[1].fpgastubid().str();
      }
      if (layerresid_[4].valid()) {
        stubid2 = layerresid_[4].fpgastubid().str();
      }
      if (layerresid_[5].valid()) {
        stubid3 = layerresid_[5].fpgastubid().str();
      }
      if (diskresid_[0].valid()) {
        stubid3 = diskresid_[0].fpgastubid().str();
      }
      if (diskresid_[1].valid()) {
        stubid2 = diskresid_[1].fpgastubid().str();
      }
    }

    if (layer() == 5) {
      if (layerresid_[0].valid()) {
        stubid0 = layerresid_[0].fpgastubid().str();
      }
      if (layerresid_[1].valid()) {
        stubid1 = layerresid_[1].fpgastubid().str();
      }
      if (layerresid_[2].valid()) {
        stubid2 = layerresid_[2].fpgastubid().str();
      }
      if (layerresid_[3].valid()) {
        stubid3 = layerresid_[3].fpgastubid().str();
      }
    }
  }

  if (isDisk()) {
    if (disk() == 1) {
      if (layerresid_[0].valid()) {
        stubid0 = layerresid_[0].fpgastubid().str();
      }
      if (diskresid_[2].valid()) {
        stubid1 = diskresid_[2].fpgastubid().str();
      }
      if (diskresid_[3].valid()) {
        stubid2 = diskresid_[3].fpgastubid().str();
      }
      if (diskresid_[4].valid()) {
        stubid3 = diskresid_[4].fpgastubid().str();
      } else if (layerresid_[1].valid()) {
        stubid3 = layerresid_[1].fpgastubid().str();
      }
    }

    if (disk() == 3) {
      if (layerresid_[0].valid()) {
        stubid0 = layerresid_[0].fpgastubid().str();
      }
      if (diskresid_[0].valid()) {
        stubid1 = diskresid_[0].fpgastubid().str();
      }
      if (diskresid_[1].valid()) {
        stubid2 = diskresid_[1].fpgastubid().str();
      }
      if (diskresid_[4].valid()) {
        stubid3 = diskresid_[4].fpgastubid().str();
      } else if (layerresid_[1].valid()) {
        stubid3 = layerresid_[1].fpgastubid().str();
      }
    }
  }

  if (isOverlap()) {
    if (layer() == 1) {
      if (diskresid_[1].valid()) {
        stubid0 = diskresid_[1].fpgastubid().str();
      }
      if (diskresid_[2].valid()) {
        stubid1 = diskresid_[2].fpgastubid().str();
      }
      if (diskresid_[3].valid()) {
        stubid2 = diskresid_[3].fpgastubid().str();
      }
      if (diskresid_[4].valid()) {
        stubid3 = diskresid_[4].fpgastubid().str();
      }
    }
  }

  std::string oss;
  // real Q print out for fitted tracks
  if (settings_.writeoutReal()) {
    oss = std::to_string((fpgafitpars_.rinv().value()) * settings_.krinvpars()) + " " +
          std::to_string((fpgafitpars_.phi0().value()) * settings_.kphi0pars()) + " " +
          std::to_string((fpgafitpars_.d0().value()) * settings_.kd0pars()) + " " +
          std::to_string((fpgafitpars_.t().value()) * settings_.ktpars()) + " " +
          std::to_string((fpgafitpars_.z0().value()) * settings_.kz()) + " " + innerFPGAStub_->phiregionaddressstr() +
          " ";
  }
  //Binary print out
  if (!settings_.writeoutReal()) {
    oss = fpgafitpars_.rinv().str() + "|" + fpgafitpars_.phi0().str() + "|" + fpgafitpars_.d0().str() + "|" +
          fpgafitpars_.t().str() + "|" + fpgafitpars_.z0().str() + "|" + innerFPGAStub_->phiregionaddressstr() + "|";
  }
  if (middleFPGAStub_) {
    oss += middleFPGAStub_->phiregionaddressstr() + " ";
  }
  oss += outerFPGAStub_->phiregionaddressstr() + " " + stubid0 + "|" + stubid1 + "|" + stubid2 + "|" + stubid3;

  return oss;
}

Track Tracklet::makeTrack(const vector<const L1TStub*>& l1stubs) {
  assert(fit());

  TrackPars<int> ipars(fpgafitpars_.rinv().value(),
                       fpgafitpars_.phi0().value(),
                       fpgafitpars_.d0().value(),
                       fpgafitpars_.t().value(),
                       fpgafitpars_.z0().value());

  Track tmpTrack(
      ipars,
      ichisqrphifit_.value(),
      ichisqrzfit_.value(),
      chisqrphifit_,
      chisqrzfit_,
      hitpattern_,
      getStubIDs(),
      (l1stubs.empty()) ? getL1Stubs() : l1stubs,  // If fitter produced no stub list, take it from original tracklet.
      getISeed());

  return tmpTrack;
}

int Tracklet::layer() const {
  int l1 = (innerFPGAStub_ && innerFPGAStub_->isBarrel()) ? innerStub_->layer() + 1 : 999,
      l2 = (middleFPGAStub_ && middleFPGAStub_->isBarrel()) ? middleStub_->layer() + 1 : 999,
      l3 = (outerFPGAStub_ && outerFPGAStub_->isBarrel()) ? outerStub_->layer() + 1 : 999, l = min(min(l1, l2), l3);
  return (l < 999 ? l : 0);
}

int Tracklet::disk() const {
  int d1 = (innerFPGAStub_ && innerFPGAStub_->isDisk()) ? innerStub_->disk() : 999,
      d2 = (middleFPGAStub_ && middleFPGAStub_->isDisk()) ? middleStub_->disk() : 999,
      d3 = (outerFPGAStub_ && outerFPGAStub_->isDisk()) ? outerStub_->disk() : 999, d = 999;
  if (abs(d1) < min(abs(d2), abs(d3)))
    d = d1;
  if (abs(d2) < min(abs(d1), abs(d3)))
    d = d2;
  if (abs(d3) < min(abs(d1), abs(d2)))
    d = d3;
  return (d < 999 ? d : 0);
}

int Tracklet::disk2() const {
  if (innerStub_->disk() > 0) {
    return innerStub_->disk() + 1;
  }
  return innerStub_->disk() - 1;
}

void Tracklet::setTrackletIndex(int index) {
  trackletIndex_ = index;
  assert(index < 128);
}

int Tracklet::getISeed() const {
  int iSeed = TCIndex_ >> 4;
  assert(iSeed >= 0 && iSeed <= (int)N_SEED);
  return iSeed;
}

int Tracklet::getITC() const {
  int iSeed = getISeed(), iTC = TCIndex_ - (iSeed << 4);
  assert(iTC >= 0 && iTC <= 14);
  return iTC;
}

unsigned int Tracklet::calcSeedIndex() const {
  int seedindex = -1;
  int seedlayer = layer();
  int seeddisk = disk();

  if (seedlayer == 1 && seeddisk == 0)
    seedindex = 0;  //L1L2
  if (seedlayer == 3 && seeddisk == 0)
    seedindex = 2;  //L3L4
  if (seedlayer == 5 && seeddisk == 0)
    seedindex = 3;  //L5L6
  if (seedlayer == 0 && abs(seeddisk) == 1)
    seedindex = 4;  //D1D2
  if (seedlayer == 0 && abs(seeddisk) == 3)
    seedindex = 5;  //D3D4
  if (seedlayer == 1 && abs(seeddisk) == 1)
    seedindex = 6;  //L1D1
  if (seedlayer == 2 && abs(seeddisk) == 1)
    seedindex = 7;  //L2D1
  if (seedlayer == 2 && abs(seeddisk) == 0)
    seedindex = 1;  //L2L3
  if (middleFPGAStub_ && seedlayer == 2 && seeddisk == 0)
    seedindex = 8;  // L3L4L2
  if (middleFPGAStub_ && seedlayer == 4 && seeddisk == 0)
    seedindex = 9;  // L5L6L4
  assert(innerFPGAStub_ != nullptr);
  assert(outerFPGAStub_ != nullptr);
  if (middleFPGAStub_ && seedlayer == 2 && abs(seeddisk) == 1) {
    int l1 = (innerFPGAStub_ && innerFPGAStub_->isBarrel()) ? innerStub_->layer() + 1 : 999,
        l2 = (middleFPGAStub_ && middleFPGAStub_->isBarrel()) ? middleStub_->layer() + 1 : 999,
        l3 = (outerFPGAStub_ && outerFPGAStub_->isBarrel()) ? outerStub_->layer() + 1 : 999;
    if (l1 + l2 + l3 < 1998) {  // If two stubs are layer stubs
      seedindex = 10;           // L2L3D1
    } else {
      seedindex = 11;  // D1D2L2
    }
  }

  if (seedindex < 0) {
    throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " seedlayer abs(seeddisk) : " << seedlayer
                                      << " " << abs(seeddisk);
  }

  return seedindex;
}
