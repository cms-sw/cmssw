#ifndef RecoParticleFlowPFClusterTools_PFResolutionMapManager_h
#define RecoParticleFlowPFClusterTools_PFResolutionMapManager_h
#include "RecoParticleFlow/PFClusterTools/interface/PFResolutionMap.h"

class PFResolutionMapManager {
 public:
  PFResolutionMapManager(const char * name);
  const PFResolutionMap& GetResolutionMap(bool MapEta,bool Corr);
  
 private:
  PFResolutionMap M1;
  PFResolutionMap M2;
  PFResolutionMap M3;
  PFResolutionMap M4;
};
#endif
