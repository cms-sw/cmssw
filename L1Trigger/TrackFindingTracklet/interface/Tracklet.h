#ifndef L1Trigger_TrackFindingTracklet_interface_Tracklet_h
#define L1Trigger_TrackFindingTracklet_interface_Tracklet_h

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <vector>
#include <memory>
#include <set>

#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"
#include "L1Trigger/TrackFindingTracklet/interface/FPGAWord.h"
#include "L1Trigger/TrackFindingTracklet/interface/Track.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackPars.h"
#include "L1Trigger/TrackFindingTracklet/interface/Projection.h"
#include "L1Trigger/TrackFindingTracklet/interface/Residual.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"

namespace trklet {

  class Settings;
  class Stub;
  class Track;

  class Tracklet {
  public:
    Tracklet(Settings const& settings,
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
             bool overlap = false);

    ~Tracklet() = default;

    //Find tp corresponding to seed.
    //Will require 'tight match' such that tp is part of each of the four clustes returns 0 if no tp matches
    int tpseed();

    bool stubtruthmatch(const L1TStub* stub);

    const Stub* innerFPGAStub() { return innerFPGAStub_; }

    const Stub* middleFPGAStub() { return middleFPGAStub_; }

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

    bool validProj(int layerdisk) const {
      assert(layerdisk >= 0 && layerdisk < N_LAYER + N_DISK);
      return proj_[layerdisk].valid();
    }

    Projection& proj(int layerdisk) {
      assert(validProj(layerdisk));
      return proj_[layerdisk];
    }

    void addMatch(unsigned int layerdisk,
                  int ideltaphi,
                  int ideltarz,
                  double dphi,
                  double drz,
                  double dphiapprox,
                  double drzapprox,
                  int stubid,
                  const trklet::Stub* stubptr);

    std::string fullmatchstr(int layer);
    std::string fullmatchdiskstr(int disk);

    bool match(unsigned int layerdisk) {
      assert(layerdisk < N_LAYER + N_DISK);
      return resid_[layerdisk].valid();
    }

    const Residual& resid(unsigned int layerdisk) {
      assert(layerdisk < N_LAYER + N_DISK);
      assert(resid_[layerdisk].valid());
      return resid_[layerdisk];
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

    // Note floating & digitized helix params after track fit.
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

    const std::string layerstubstr(const unsigned layer) const;
    const std::string diskstubstr(const unsigned disk) const;
    std::string trackfitstr() const;

    // Create a Track object from stubs & digitized track helix params
    Track makeTrack(const std::vector<const L1TStub*>& l1stubs);

    Track* getTrack() {
      assert(fpgatrack_ != nullptr);
      return fpgatrack_.get();
    }

    bool fit() const { return ichisqrphifit_.value() != -1; }

    int layer() const;
    int disk() const;

    bool isBarrel() const { return barrel_; }
    bool isOverlap() const { return overlap_; }
    int isDisk() const { return disk_; }

    void setTrackletIndex(unsigned int index);

    int trackletIndex() const { return trackletIndex_; }

    void setTCIndex(int index) { TCIndex_ = index; }

    int TCIndex() const { return TCIndex_; }

    int TCID() const { return TCIndex_ * (1 << settings_.nbitstrackletindex()) + trackletIndex_; }

    int getISeed() const;
    int getITC() const;

    void setTrackIndex(int index);
    int trackIndex() const;

    unsigned int PSseed() const { return ((layer() == 1) || (layer() == 2) || (disk() != 0)) ? 1 : 0; }

    unsigned int seedIndex() const { return seedIndex_; }

  private:
    unsigned int seedIndex_;

    // three types of tracklets + one triplet
    bool barrel_;
    bool disk_;
    bool overlap_;
    bool triplet_;

    const Stub* innerFPGAStub_;
    const Stub* middleFPGAStub_;
    const Stub* outerFPGAStub_;

    int trackletIndex_;
    int TCIndex_;
    int trackIndex_;

    //Tracklet track parameters
    TrackPars<FPGAWord> fpgapars_;

    TrackPars<double> trackpars_;
    TrackPars<double> trackparsapprox_;

    // the layer/disk ids that we project to (never project to >4 barrel layers)
    int projlayer_[N_LAYER - 2];
    int projdisk_[N_DISK];

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

    Projection proj_[N_LAYER + N_DISK];

    Residual resid_[N_LAYER + N_DISK];

    Settings const& settings_;
  };
};  // namespace trklet
#endif
