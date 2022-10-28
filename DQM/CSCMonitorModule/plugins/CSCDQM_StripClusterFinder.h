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
    int32_t LId;
    uint32_t i;
    uint32_t j;
    uint32_t ic1, IC1MIN, IC1MAX, JC1MIN, JC1MAX, ic2, IC2MIN, IC2MAX, JC2MIN, JC2MAX, icstart;
    int LayerNmb, TimeSliceNmb, StripNmb, AnodeGroupNmb, AFEBSliceNmb;
    void SearchMax(void);
    void SearchBorders(void);
    void Match(void);
    bool FindAndMatch(void);
    void KillCluster(void);
    void RefindMax(void);
    bool is7DCFEBs;
    bool isME11;
  };

}  // namespace cscdqm

#endif
