#ifndef L1Trigger_TrackFindingTracklet_interface_ProjectionRouterBendTable_h
#define L1Trigger_TrackFindingTracklet_interface_ProjectionRouterBendTable_h

#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <vector>

namespace trklet {

  class Globals;

  class ProjectionRouterBendTable {
  public:
    ProjectionRouterBendTable() {}

    ~ProjectionRouterBendTable() = default;

    void init(Settings const& settings, Globals* globals, unsigned int nrbits, unsigned int nphiderbits);

    int bendLoookup(int diskindex, int bendindex);

  private:
    std::vector<int> bendtable_[N_DISK];
  };

};  // namespace trklet
#endif
