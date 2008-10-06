#ifndef CSCDQM_CSCStripClusterFinder_h
#define CSCDQM_CSCStripClusterFinder_h

#include "DQM/CSCMonitorModule/interface/CSCDQM_CSCStripCluster.h"

#include <vector>
#include <iostream>
#include <string>
#include <signal.h>
#include <map>
#include <string>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdint.h>

namespace cscdqm {

using namespace std;

class CSCStripClusterFinder
{
 public:
  CSCStripClusterFinder(int l, int s, int cf, int st );
  void DoAction(int layerId,float *cathodes);
  void printClusters(void);
  std::vector<CSCStripClusterFitData> thePulseHeightMap;
 public:
  class Sort{
  public:
    bool  operator()(CSCStripClusterFitData a,CSCStripClusterFitData b) const;
  };
  std::vector<CSCStripCluster> MEStripClusters;
  ClusterLocalMax localMaxTMP;
  std::vector<CSCStripCluster> getClusters(){ return MEStripClusters;}
 private:
  int32_t LId;
  uint32_t i;
  uint32_t j;
  uint32_t ic1,IC1MIN,IC1MAX,JC1MIN,JC1MAX,ic2,IC2MIN,IC2MAX,JC2MIN,JC2MAX,icstart;
  int LayerNmb, TimeSliceNmb, StripNmb, AnodeGroupNmb,AFEBSliceNmb; 
  void SearchMax(void);
  void SearchBorders(void);
  void Match(void);
  bool FindAndMatch(void);
  void KillCluster(void);
  void RefindMax(void);
  
};

}

#endif

