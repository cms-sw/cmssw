#ifndef CSCDQM_StripCluster_h
#define CSCDQM_StripCluster_h

#include <TObject.h>
#include <vector>
#include <algorithm>

#include "CSCDQM_StripClusterFitData.h"
#include "CSCDQM_ClusterLocalMax.h"

namespace cscdqm {

  /**
   * @class StripCluster
   * @brief Strip Cluster
   */
class StripCluster {
 public:
  std::vector<StripClusterFitData> ClusterPulseMapHeight;
  std::vector<ClusterLocalMax> localMax;
  int LFTBNDTime;
  int LFTBNDStrip;
  int IRTBNDTime;
  int IRTBNDStrip;
  int LayerId;
  int EventId;
  float Mean[2];

  int rlocalMaxTime(int i){return localMax[i].Time;}
  int rlocalMaxStrip(int i){return localMax[i].Strip;}
  int rLFTBNDTime(void){return LFTBNDTime;}
  int rLFTBNDStrip(void){return LFTBNDStrip;}
  int rIRTBNDTime(void){return IRTBNDTime;}
  int rIRTBNDStrip(void){return IRTBNDStrip;}	
  int rnlocal(){return localMax.size();}
  StripCluster();
  virtual ~StripCluster();
//  ClassDef(StripCluster,1) //StripCluster

};

}

#endif
