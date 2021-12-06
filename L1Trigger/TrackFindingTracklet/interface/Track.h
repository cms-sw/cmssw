#ifndef L1Trigger_TrackFindingTracklet_interface_Track_h
#define L1Trigger_TrackFindingTracklet_interface_Track_h

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <vector>
#include <map>

#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"
#include "L1Trigger/TrackFindingTracklet/interface/SLHCEvent.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackPars.h"

namespace trklet {

  class Track {
  public:
    Track(TrackPars<int> ipars,
          int ichisqrphi,
          int ichisqrz,
          double chisqrphi,
          double chisqrz,
          int hitpattern,
          std::map<int, int> stubID,
          const std::vector<L1TStub>& l1stub,
          int seed);

    ~Track() = default;

    void setDuplicate(bool flag) { duplicate_ = flag; }
    void setSector(int nsec) { sector_ = nsec; }
    void setStubIDpremerge(std::vector<std::pair<int, int>> stubIDpremerge) { stubIDpremerge_ = stubIDpremerge; }
    void setStubIDprefit(std::vector<std::pair<int, int>> stubIDprefit) { stubIDprefit_ = stubIDprefit; }

    const TrackPars<int>& pars() const { return ipars_; }

    int ichisq() const { return ichisqrphi_ + ichisqrz_; }

    const std::map<int, int>& stubID() const { return stubID_; }
    const std::vector<L1TStub>& stubs() const { return l1stub_; }

    //These are not used? Should be removed?
    const std::vector<std::pair<int, int>>& stubIDpremerge() const { return stubIDpremerge_; }
    const std::vector<std::pair<int, int>>& stubIDprefit() const { return stubIDprefit_; }

    int hitpattern() const { return hitpattern_; }
    int seed() const { return seed_; }
    int duplicate() const { return duplicate_; }
    int sector() const { return sector_; }

    double pt(Settings const& settings) const {
      return (settings.c() * settings.bfield() * 0.01) / (ipars_.rinv() * settings.krinvpars());
    }

    double phi0(Settings const& settings) const;

    double eta(Settings const& settings) const { return asinh(ipars_.t() * settings.ktpars()); }
    double tanL(Settings const& settings) const { return ipars_.t() * settings.ktpars(); }
    double z0(Settings const& settings) const { return ipars_.z0() * settings.kz0pars(); }
    double rinv(Settings const& settings) const { return ipars_.rinv() * settings.krinvpars(); }
    double d0(Settings const& settings) const { return ipars_.d0() * settings.kd0pars(); }
    double chisq() const { return chisqrphi_ + chisqrz_; }

    double chisqrphi() const { return chisqrphi_; }
    double chisqrz() const { return chisqrz_; }

    int nPSstubs() const {
      int npsstubs = 0;
      for (const auto& istub : l1stub_) {
        if (istub.layer() < N_PSLAYER)
          npsstubs++;
      }
      return npsstubs;
    }

  private:
    TrackPars<int> ipars_;
    int ichisqrphi_;
    int ichisqrz_;

    double chisqrphi_;
    double chisqrz_;

    int hitpattern_;

    std::vector<std::pair<int, int>> stubIDpremerge_;
    std::vector<std::pair<int, int>> stubIDprefit_;
    std::map<int, int> stubID_;
    std::vector<L1TStub> l1stub_;

    unsigned int nstubs_;
    int seed_;
    bool duplicate_;
    int sector_;
  };

};  // namespace trklet
#endif
