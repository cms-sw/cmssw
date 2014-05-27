#ifndef _Arbor_hh_
#define _Arbor_hh_

#include <string>
#include <iostream>
#include <TVector3.h>
#include "ArborTool.hh"

void init(float CellSize, float LayerThickness);

void HitsCleaning( std::vector<TVector3> inputHits );

void HitsClassification( linkcoll inputLinks );

void BuildInitLink(float Threshold);

void LinkIteration(float Threshold);

void BranchBuilding();

void BushMerging();

void BushAbsorbing();

void MakingCMSCluster();

branchcoll Arbor( std::vector<TVector3>, float CellSize, float LayerThickness );

#endif


