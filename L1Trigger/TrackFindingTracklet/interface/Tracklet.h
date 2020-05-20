#ifndef L1Trigger_TrackFindingTracklet_interface_Tracklet_h
#define L1Trigger_TrackFindingTracklet_interface_Tracklet_h

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <vector>
#include <set>

#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"
#include "L1Trigger/TrackFindingTracklet/interface/FPGAWord.h"
#include "L1Trigger/TrackFindingTracklet/interface/Track.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackPars.h"
#include "L1Trigger/TrackFindingTracklet/interface/LayerProjection.h"
#include "L1Trigger/TrackFindingTracklet/interface/DiskProjection.h"
#include "L1Trigger/TrackFindingTracklet/interface/LayerResidual.h"
#include "L1Trigger/TrackFindingTracklet/interface/DiskResidual.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"

namespace trklet {

  class Settings;
  class Stub;
  class Track;

  class Tracklet {
  public:
    Tracklet(const Settings* settings,
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
             bool overlap = false);

    ~Tracklet() = default;

    //Find tp corresponding to seed.
    //Will require 'tight match' such that tp is part of each of the four clustes returns 0 if no tp matches
    int tpseed();

    bool stubtruthmatch(const L1TStub* stub);

    const L1TStub* innerStub() { return innerStub_; }
    const Stub* innerFPGAStub() { return innerFPGAStub_; }

    const L1TStub* middleStub() { return middleStub_; }
    const Stub* middleFPGAStub() { return middleFPGAStub_; }

    const L1TStub* outerStub() { return outerStub_; }
    const Stub* outerFPGAStub() { return outerFPGAStub_; }

    std::string addressstr();

    //Tracklet parameters print out
    std::string trackletparstr();

    std::string vmstrlayer(int layer, unsigned int allstubindex);

    std::string vmstrdisk(int disk, unsigned int allstubindex);

    std::string trackletprojstr(int layer) const;
    std::string trackletprojstrD(int disk) const;

    std::string trackletprojstrlayer(int layer) const { return trackletprojstr(layer); }
    std::string trackletprojstrdisk(int disk) const { return trackletprojstrD(disk); }

    bool validProj(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerproj_[layer - 1].valid();
    }

    const FPGAWord& fpgaphiprojder(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerproj_[layer - 1].fpgaphiprojder();
    }

    const FPGAWord& fpgazproj(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerproj_[layer - 1].fpgazproj();
    }

    const FPGAWord& fpgaphiproj(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerproj_[layer - 1].fpgaphiproj();
    }

    const FPGAWord& fpgazprojder(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerproj_[layer - 1].fpgazprojder();
    }

    int zbin1projvm(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerproj_[layer - 1].fpgazbin1projvm().value();
    }

    int zbin2projvm(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerproj_[layer - 1].fpgazbin2projvm().value();
    }

    int finezvm(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerproj_[layer - 1].fpgafinezvm().value();
    }

    int rbin1projvm(int disk) const {
      assert(disk > 0 && disk <= N_DISK);
      return diskproj_[disk - 1].fpgarbin1projvm().value();
    }

    int rbin2projvm(int disk) const {
      assert(disk > 0 && disk <= N_DISK);
      return diskproj_[disk - 1].fpgarbin2projvm().value();
    }

    int finervm(int disk) const {
      assert(disk > 0 && disk <= N_DISK);
      return diskproj_[disk - 1].fpgafinervm().value();
    }

    int phiprojvm(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerproj_[layer - 1].fpgaphiprojvm().value();
    }

    int zprojvm(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerproj_[layer - 1].fpgazprojvm().value();
    }

    double phiproj(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerproj_[layer - 1].phiproj();
    }

    double phiprojder(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerproj_[layer - 1].phiprojder();
    }

    double zproj(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerproj_[layer - 1].zproj();
    }

    double zprojder(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerproj_[layer - 1].zprojder();
    }

    double zprojapprox(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerproj_[layer - 1].zprojapprox();
    }

    double zprojderapprox(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerproj_[layer - 1].zprojderapprox();
    }

    double phiprojapprox(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerproj_[layer - 1].phiprojapprox();
    }

    double phiprojderapprox(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerproj_[layer - 1].phiprojderapprox();
    }

    double rproj(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerproj_[layer - 1].rproj();
    }

    double rstub(int layer) {
      assert(layer > 0 && layer <= N_LAYER);
      return layerresid_[layer - 1].rstub();
    }

    //Disks residuals

    bool validProjDisk(int disk) const {
      assert(abs(disk) <= N_DISK);
      return diskproj_[abs(disk) - 1].valid();
    }

    const FPGAWord& fpgaphiresiddisk(int disk) {
      assert(abs(disk) <= N_DISK);
      return diskresid_[abs(disk) - 1].fpgaphiresid();
    }

    const FPGAWord& fpgarresiddisk(int disk) {
      assert(abs(disk) <= N_DISK);
      return diskresid_[abs(disk) - 1].fpgarresid();
    }

    double phiresiddisk(int disk) {
      assert(abs(disk) <= N_DISK);
      return diskresid_[abs(disk) - 1].phiresid();
    }

    double rresiddisk(int disk) {
      assert(abs(disk) <= N_DISK);
      return diskresid_[abs(disk) - 1].rresid();
    }

    double phiresidapproxdisk(int disk) {
      assert(abs(disk) <= N_DISK);
      return diskresid_[abs(disk) - 1].phiresidapprox();
    }

    double rresidapproxdisk(int disk) {
      assert(abs(disk) <= N_DISK);
      return diskresid_[abs(disk) - 1].rresidapprox();
    }

    double zstubdisk(int disk) {
      assert(abs(disk) <= N_DISK);
      return diskresid_[abs(disk) - 1].zstub();
    }

    void setBendIndex(int bendIndex, int disk) {
      assert(abs(disk) <= N_DISK);
      diskproj_[abs(disk) - 1].setBendIndex(bendIndex);
    }

    const FPGAWord& getBendIndex(int disk) const {
      assert(abs(disk) <= N_DISK);
      return diskproj_[abs(disk) - 1].getBendIndex();
    }

    double alphadisk(int disk) const {
      assert(abs(disk) <= N_DISK);
      return diskresid_[abs(disk) - 1].alpha();
    }

    const FPGAWord& ialphadisk(int disk) const {
      assert(abs(disk) <= N_DISK);
      return diskresid_[abs(disk) - 1].ialpha();
    }

    const FPGAWord& fpgaphiprojdisk(int disk) const {
      assert(abs(disk) <= N_DISK);
      return diskproj_[abs(disk) - 1].fpgaphiproj();
    }

    const FPGAWord& fpgaphiprojderdisk(int disk) const {
      assert(abs(disk) <= N_DISK);
      return diskproj_[abs(disk) - 1].fpgaphiprojder();
    }

    const FPGAWord& fpgarprojdisk(int disk) const {
      assert(abs(disk) <= N_DISK);
      return diskproj_[abs(disk) - 1].fpgarproj();
    }

    const FPGAWord& fpgarprojderdisk(int disk) const {
      assert(abs(disk) <= N_DISK);
      return diskproj_[abs(disk) - 1].fpgarprojder();
    }

    double phiprojapproxdisk(int disk) const {
      assert(abs(disk) <= N_DISK);
      return diskproj_[abs(disk) - 1].phiprojapprox();
    }

    double phiprojderapproxdisk(int disk) const {
      assert(abs(disk) <= N_DISK);
      return diskproj_[abs(disk) - 1].phiprojderapprox();
    }

    double rprojapproxdisk(int disk) const {
      assert(abs(disk) <= N_DISK);
      return diskproj_[abs(disk) - 1].rprojapprox();
    }

    double rprojderapproxdisk(int disk) const {
      assert(abs(disk) <= N_DISK);
      return diskproj_[abs(disk) - 1].rprojderapprox();
    }

    double phiprojdisk(int disk) const {
      assert(abs(disk) <= N_DISK);
      return diskproj_[abs(disk) - 1].phiproj();
    }

    double phiprojderdisk(int disk) const {
      assert(abs(disk) <= N_DISK);
      return diskproj_[abs(disk) - 1].phiprojder();
    }

    double rprojdisk(int disk) const {
      assert(abs(disk) <= N_DISK);
      return diskproj_[abs(disk) - 1].rproj();
    }

    double rprojderdisk(int disk) const {
      assert(abs(disk) <= N_DISK);
      return diskproj_[abs(disk) - 1].rprojder();
    }

    bool matchdisk(int disk) {
      assert(abs(disk) <= N_DISK);
      return diskresid_[abs(disk) - 1].valid();
    }

    void addMatch(int layer,
                  int ideltaphi,
                  int ideltaz,
                  double dphi,
                  double dz,
                  double dphiapprox,
                  double dzapprox,
                  int stubid,
                  double rstub,
                  const trklet::Stub* stubptr);

    void addMatchDisk(int disk,
                      int ideltaphi,
                      int ideltar,
                      double dphi,
                      double dr,
                      double dphiapprox,
                      double drapprox,
                      double alpha,
                      int stubid,
                      double zstub,
                      const trklet::Stub* stubptr);

    int nMatches();
    int nMatchesDisk();

    bool match(int layer) {
      assert(layer > 0 && layer <= N_LAYER);
      return layerresid_[layer - 1].valid();
    }

    std::string fullmatchstr(int layer);
    std::string fullmatchdiskstr(int disk);

    bool validResid(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerresid_[layer - 1].valid();
    }

    const trklet::Stub* stubptr(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerresid_[layer - 1].stubptr();
    }

    double phiresid(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerresid_[layer - 1].phiresid();
    }

    double phiresidapprox(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerresid_[layer - 1].phiresidapprox();
    }

    double zresid(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerresid_[layer - 1].zresid();
    }

    double zresidapprox(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerresid_[layer - 1].zresidapprox();
    }

    const FPGAWord& fpgaphiresid(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerresid_[layer - 1].fpgaphiresid();
    }

    const FPGAWord& fpgazresid(int layer) const {
      assert(layer > 0 && layer <= N_LAYER);
      return layerresid_[layer - 1].fpgazresid();
    }

    std::vector<const L1TStub*> getL1Stubs();

    std::map<int, int> getStubIDs();

    double rinv() const { return trackpars_.rinv(); }
    double phi0() const { return trackpars_.phi0(); }
    double d0() const { return trackpars_.d0(); }
    double t() const { return trackpars_.t(); }
    double z0() const { return trackpars_.z0(); }

    double rinvapprox() const { return trackparsapprox_.rinv(); }
    double phi0approx() const { return trackparsapprox_.phi0(); }
    double d0approx() const { return trackparsapprox_.d0(); }
    double tapprox() const { return trackparsapprox_.t(); }
    double z0approx() const { return trackparsapprox_.z0(); }

    const FPGAWord& fpgarinv() const { return fpgapars_.rinv(); }
    const FPGAWord& fpgaphi0() const { return fpgapars_.phi0(); }
    const FPGAWord& fpgad0() const { return fpgapars_.d0(); }
    const FPGAWord& fpgat() const { return fpgapars_.t(); }
    const FPGAWord& fpgaz0() const { return fpgapars_.z0(); }

    double rinvfit() const { return fitpars_.rinv(); }
    double phi0fit() const { return fitpars_.phi0(); }
    double d0fit() const { return fitpars_.d0(); }
    double tfit() const { return fitpars_.t(); }
    double z0fit() const { return fitpars_.z0(); }
    double chiSqfit() const { return chisqrphifit_ + chisqrzfit_; }

    double rinvfitexact() const { return fitparsexact_.rinv(); }
    double phi0fitexact() const { return fitparsexact_.phi0(); }
    double d0fitexact() const { return fitparsexact_.d0(); }
    double tfitexact() const { return fitparsexact_.t(); }
    double z0fitexact() const { return fitparsexact_.z0(); }

    const FPGAWord& irinvfit() const { return fpgafitpars_.rinv(); }
    const FPGAWord& iphi0fit() const { return fpgafitpars_.phi0(); }
    const FPGAWord& id0fit() const { return fpgafitpars_.d0(); }
    const FPGAWord& itfit() const { return fpgafitpars_.t(); }
    const FPGAWord& iz0fit() const { return fpgafitpars_.z0(); }
    FPGAWord ichiSqfit() const {
      return FPGAWord(ichisqrphifit_.value() + ichisqrzfit_.value(), ichisqrphifit_.nbits());
    }

    void setFitPars(double rinvfit,
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
                    const std::vector<const L1TStub*>& l1stubs = std::vector<const L1TStub*>());

    std::string trackfitstr();

    Track makeTrack(const std::vector<const L1TStub*>& l1stubs);

    Track* getTrack() {
      assert(fpgatrack_ != nullptr);
      return fpgatrack_.get();
    }

    bool fit() const { return ichisqrphifit_.value() != -1; }

    int layer() const;
    int disk() const;
    int disk2() const;

    bool isBarrel() const { return barrel_; }
    bool isOverlap() const { return overlap_; }
    int isDisk() const { return disk_; }

    void setTrackletIndex(int index);

    int trackletIndex() const { return trackletIndex_; }

    void setTCIndex(int index) { TCIndex_ = index; }

    int TCIndex() const { return TCIndex_; }

    int TCID() const { return TCIndex_ * (1 << 7) + trackletIndex_; }

    int getISeed() const;
    int getITC() const;

    unsigned int PSseed() const { return ((layer() == 1) || (layer() == 2) || (disk() != 0)) ? 1 : 0; }

    unsigned int seedIndex() const { return seedIndex_; }

    unsigned int calcSeedIndex() const;

  private:
    unsigned int seedIndex_;

    // three types of trackletss
    bool barrel_;
    bool disk_;
    bool overlap_;
    bool triplet_;

    const Stub* innerFPGAStub_;
    const Stub* middleFPGAStub_;
    const Stub* outerFPGAStub_;

    const L1TStub* innerStub_;
    const L1TStub* middleStub_;
    const L1TStub* outerStub_;

    int trackletIndex_;
    int TCIndex_;

    //Tracklet track parameters
    TrackPars<FPGAWord> fpgapars_;

    TrackPars<double> trackpars_;
    TrackPars<double> trackparsapprox_;

    int projlayer_[N_PROJLAYER];
    int projdisk_[N_PROJDISK];

    //Track parameters from track fit
    TrackPars<FPGAWord> fpgafitpars_;
    FPGAWord ichisqrphifit_;
    FPGAWord ichisqrzfit_;

    TrackPars<double> fitpars_;
    double chisqrphifit_;
    double chisqrzfit_;

    TrackPars<double> fitparsexact_;
    double chisqrphifitexact_;
    double chisqrzfitexact_;

    int hitpattern_;

    std::unique_ptr<Track> fpgatrack_;

    LayerProjection layerproj_[N_LAYER];
    DiskProjection diskproj_[N_DISK];

    LayerResidual layerresid_[N_LAYER];
    DiskResidual diskresid_[N_DISK];

    const Settings* settings_;
  };
};  // namespace trklet
#endif
