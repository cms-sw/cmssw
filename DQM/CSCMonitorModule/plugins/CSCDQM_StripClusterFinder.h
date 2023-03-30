#ifndef CSCDQM_StripClusterFinder_h
#define CSCDQM_StripClusterFinder_h

#include "CSCDQM_StripCluster.h"

#include <vector>
#include <iostream>
#include <string>
#include <csignal>
#include <map>
#include <string>
#include <iomanip>
#include <set>
#include <sstream>
#include <cstdint>

namespace cscdqm {

  /**
 * @class StripClusterFinder
 * @brief Object used to find Strip Clusters
 */
  class StripClusterFinder {
  public:
    StripClusterFinder(int l, int s, int cf, int st, bool ME11 = false);
    void DoAction(int layerId, float* cathodes);
    void printClusters(void);
    std::vector<StripClusterFitData> thePulseHeightMap;

  public:
    class Sort {
    public:
      bool operator()(const StripClusterFitData& a, const StripClusterFitData& b) const;
    };
    std::vector<StripCluster> MEStripClusters;
    ClusterLocalMax localMaxTMP;
    std::vector<StripCluster> getClusters() { return MEStripClusters; }

  private:
    struct C1 {
      uint32_t IC1MIN, IC1MAX, JC1MIN, JC1MAX;
    };
    struct C2 {
      uint32_t IC2MIN, IC2MAX, JC2MIN, JC2MAX;
    };
    int LayerNmb, TimeSliceNmb, StripNmb;
    void SearchMax(int32_t layerId);
    void SearchBorders(void);
    void Match(void);
    bool FindAndMatch(void);
    void KillCluster(uint32_t ic1, uint32_t ic2, C1 const&, C2 const&);
    void RefindMax(void);
    bool is7DCFEBs;
    bool isME11;
  };

}  // namespace cscdqm

#endif
