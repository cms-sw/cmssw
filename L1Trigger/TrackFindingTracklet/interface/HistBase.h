#ifndef L1Trigger_TrackFindingTracklet_interface_HistBase_h
#define L1Trigger_TrackFindingTracklet_interface_HistBase_h

#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <bitset>
#include <cassert>
#include <cmath>

namespace trklet {
  class Globals;

  class HistBase {
  public:
    HistBase() {}

    virtual ~HistBase() = default;

    virtual void open() {}
    virtual void close() {}

    virtual void bookLayerResidual() {}
    virtual void bookDiskResidual() {}
    virtual void bookTrackletParams() {}
    virtual void bookSeedEff() {}

    virtual void FillLayerResidual(int, int, double, double, double, double, bool) {}

    virtual void FillDiskResidual(int, int, double, double, double, double, bool) {}

    //arguments are
    // int seedIndex
    // int iSector
    // double irinv, rinv
    // double iphi0, phi0
    // double ieta, eta
    // double iz0, z0
    // int tp
    virtual void fillTrackletParams(
        const Settings*, Globals*, int, int, double, double, double, double, double, double, double, double, int) {}

    //int seedIndex
    //double etaTP
    //bool eff
    virtual void fillSeedEff(int, double, bool) {}

  private:
  };

};  // namespace trklet
#endif
