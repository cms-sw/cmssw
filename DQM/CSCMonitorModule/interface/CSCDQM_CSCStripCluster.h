#ifndef CSCDQM_CSCStripCluster_h
#define CSCDQM_CSCStripCluster_h

#include <TObject.h>
#include <vector>
#include <algorithm>

#include "DQM/CSCMonitorModule/interface/CSCDQM_CSCStripClusterFitData.h"
#include "DQM/CSCMonitorModule/interface/CSCDQM_ClusterLocalMax.h"

namespace cscdqm {

class CSCStripCluster {
 public:
  std::vector<CSCStripClusterFitData> ClusterPulseMapHeight;
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
  CSCStripCluster();
  virtual ~CSCStripCluster();
//  ClassDef(CSCStripCluster,1) //CSCStripCluster

};

}

#endif
