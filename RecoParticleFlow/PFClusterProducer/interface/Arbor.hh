#ifndef _Arbor_hh_
#define _Arbor_hh_

#include <string>
#include <iostream>
#include <TVector3.h>
#include "ArborTool.hh"

namespace arbor {

  void init(float CellSize, float LayerThickness);
  
  void HitsCleaning(std::vector<std::pair<TVector3,float> > inputHits );
  
  void HitsClassification( linkcoll inputLinks );
  
  void BuildInitLink(float Threshold);
  
  void LinkIteration(float Threshold);
  
  void BranchBuilding(const float distSeedForMerge, const bool allowSameLayerSeedMerge);
  
  void BushMerging();
  
  void BushAbsorbing();
  
  void MakingCMSCluster();
  
  branchcoll Arbor( std::vector<std::pair<TVector3,float> >, const float CellSize, const float LayerThickness, const float distSeedForMerge, const bool allowSameLayerSeedMerge );
}

#endif


