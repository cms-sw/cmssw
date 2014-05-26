#ifndef ARBORTOOL_H_
#define ARBORTOOL_H_

#include "TVector3.h"
#include "TMatrixF.h"
#include <iostream>
#include <vector>
#include <string>

typedef std::vector< std::vector<int> > branchcoll;
typedef std::vector<int> branch;
typedef std::vector< std::pair<int, int> > linkcoll;

int BarrelFlag( TVector3 inputPos );

int DepthFlag( TVector3 inputPos );

TVector3 CalVertex( TVector3 Pos1, TVector3 Dir1, TVector3 Pos2, TVector3 Dir2 );

int TPCPosition( TVector3 inputPos );		//Used to tag MCParticle position, if generated inside TPC & Dead outside

float DisSeedSurface( TVector3 SeedPos );	//for a given position, calculate the distance to Calo surface ( ECAL )

float DisTPCBoundary( TVector3 Pos );

TMatrixF MatrixSummarize( TMatrixF inputMatrix );

std::vector<int>SortMeasure( std::vector<float> Measure, int ControlOrderFlag );

float DistanceChargedParticleToCluster(TVector3 CPRefPos, TVector3 CPRefMom, TVector3 CluPosition);

branchcoll ArborBranchMerge(branchcoll inputbranches, TMatrixF inputMatrix);

#endif //
