#ifndef L1Trigger_TrackFindingTracklet_interface_ProjectionRouterBendTable_h
#define L1Trigger_TrackFindingTracklet_interface_ProjectionRouterBendTable_h

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <vector>

namespace trklet {

  class Settings;
  class Globals;

  class ProjectionRouterBendTable {
  public:
    ProjectionRouterBendTable() {}

    ~ProjectionRouterBendTable() = default;

    void init(const Settings* settings, Globals* globals, unsigned int nrbits, unsigned int nphiderbits);

    int bendLoookup(int diskindex, int bendindex);

  private:
    std::vector<int> bendtable_[5];
  };

};  // namespace trklet
#endif
