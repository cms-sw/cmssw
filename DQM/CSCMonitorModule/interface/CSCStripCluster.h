#ifndef CSCStripCluster_h
#define CSCStripCluster_h

#include "DQM/CSCMonitorModule/interface/CSCStripClusterFitData.h"
#include "DQM/CSCMonitorModule/interface/ClusterLocalMax.h"
#include <TObject.h>
#include <vector>
#include <algorithm>

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
#endif
