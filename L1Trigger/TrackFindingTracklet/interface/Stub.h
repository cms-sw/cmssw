#ifndef L1Trigger_TrackFindingTracklet_interface_Stub_h
#define L1Trigger_TrackFindingTracklet_interface_Stub_h

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>

#include "L1Trigger/TrackFindingTracklet/interface/FPGAWord.h"
#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

namespace trklet {

  class Globals;

  class Stub {
  public:
    Stub(Settings const& settings);

    Stub(L1TStub& stub, Settings const& settings, Globals& globals);

    ~Stub() = default;

    FPGAWord iphivmFineBins(int VMbits, int finebits) const;

    std::string str() const {
      if (layer_.value() != -1) {
        return r_.str() + "|" + z_.str() + "|" + phi_.str() + "|" + bend_.str();
      } else {
        if (isPSmodule()) {
          return r_.str() + "|" + z_.str() + "|" + phi_.str() + "|" + bend_.str();
        } else {
          return "000" + r_.str() + "|" + z_.str() + "|" + phi_.str() + "|" + alpha_.str() + "|" + bend_.str();
        }
      }
    }

    std::string strbare() const { return bend_.str() + r_.str() + z_.str() + phi_.str(); }

    std::string strinner() const {
      unsigned int nbitsfinephi = 8;
      FPGAWord finephi(
          phicorr_.bits(phicorr_.nbits() - nbitsfinephi, nbitsfinephi), nbitsfinephi, true, __LINE__, __FILE__);
      if (layer_.value() == -1) {
        return str() + "|" + negdisk_.str() + "|" + stubindex_.str() + "|" + finephi.str();
      } else {
        return str() + "|" + stubindex_.str() + "|" + finephi.str();
      }
    }

    FPGAWord allStubIndex() const { return stubindex_; }

    unsigned int phiregionaddress() const;
    std::string phiregionaddressstr() const;

    void setAllStubIndex(int nstub);  //should migrate away from using this method

    void setPhiCorr(int phiCorr);

    const FPGAWord& bend() const { return bend_; }

    const FPGAWord& r() const { return r_; }
    const FPGAWord& z() const { return z_; }
    const FPGAWord& negdisk() const { return negdisk_; }
    const FPGAWord& phi() const { return phi_; }
    const FPGAWord& phicorr() const { return phicorr_; }
    const FPGAWord& alpha() const { return alpha_; }

    const FPGAWord& stubindex() const { return stubindex_; }
    const FPGAWord& layer() const { return layer_; }
    const FPGAWord& disk() const { return disk_; }
    unsigned int layerdisk() const;

    bool isPSmodule() const { return (layerdisk_ < N_LAYER) ? (layerdisk_ < N_PSLAYER) : (r_.value() > 10); }

    double rapprox() const;
    double zapprox() const;
    double phiapprox(double phimin, double) const;

    L1TStub* l1tstub() { return l1tstub_; }
    const L1TStub* l1tstub() const { return l1tstub_; }
    void setl1tstub(L1TStub* l1tstub) { l1tstub_ = l1tstub; }

    bool isBarrel() const { return layerdisk_ < N_LAYER; }

  private:
    unsigned int layerdisk_;

    FPGAWord layer_;
    FPGAWord disk_;
    FPGAWord r_;
    FPGAWord z_;
    FPGAWord negdisk_;
    FPGAWord phi_;
    FPGAWord alpha_;

    FPGAWord bend_;

    FPGAWord phicorr_;  //Corrected for bend to nominal radius

    FPGAWord stubindex_;

    L1TStub* l1tstub_;
    Settings const& settings_;
  };

};  // namespace trklet
#endif
