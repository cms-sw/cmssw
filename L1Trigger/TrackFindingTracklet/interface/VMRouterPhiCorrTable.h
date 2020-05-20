#ifndef L1Trigger_TrackFindingTracklet_interface_VMRouterPhiCorrTable_h
#define L1Trigger_TrackFindingTracklet_interface_VMRouterPhiCorrTable_h

#include "L1Trigger/TrackFindingTracklet/interface/TETableBase.h"

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <vector>

namespace trklet {

  class Settings;

  class VMRouterPhiCorrTable : public TETableBase {
  public:
    VMRouterPhiCorrTable();

    ~VMRouterPhiCorrTable() override = default;

    void init(const Settings* settings, int layer, int bendbits, int rbits);

    int getphiCorrValue(int ibend, int irbin) const;

    int lookupPhiCorr(int ibend, int rbin);

  private:
    double rmean_;
    double rmin_;
    double rmax_;

    double dr_;

    int bendbits_;
    int rbits_;

    int bendbins_;
    int rbins_;

    int layer_;
  };
};  // namespace trklet
#endif
