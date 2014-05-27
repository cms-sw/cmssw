#include "RecoParticleFlow/PFClusterProducer/plugins/ArborTool.hh"
#include <TMath.h>

using namespace std;

const double pi = acos(-1.0); 

//Geometric Parameter - ... need to be changed for different detector models

float DHCALBarrelLength = 2350.0;				
const float DHCALBarrelRadius = 2058.0;         //Octo
const float DHCALBarrelOuterR = 3381.6;
const float DHCALEndCapInnerZ = 2650.0;
const float DHCALEndCapOuterZ = 3970.0;         // Should be ... 3922
const float ECALBarrelRadius = 1843.0;		
const float ECALBarrelLength = 2350.0;          // identical to DHCALBarrelLength 
const float ECALBarrelOuterR = 2028.0;
const float ECALEndCapInnerZ = 2450.0;
const float ECALEndCapOuterZ = 2635.0;
const float ECALEndCapOuterR = 2088.0;
const float ECALHalfZ = 2350.0; // mm, Endcap Ente
const float ECALRadius = 1847.4; // mm... minimal part for the octagnle.




int BarrelFlag( TVector3 inputPos)
{
	int isBarrel = 0;	

	if(fabs(inputPos[2]) < ECALBarrelLength)
	{
		isBarrel = 1; 
	}

	return isBarrel; 
}

int DepthFlag( TVector3 inputPos )      //Used to calculate depth of given position...
{

	float ShiftDHCAL = 530; // 20 Layers Inside DHCAL

	float DHCALDeepInnerZ = DHCALEndCapInnerZ + ShiftDHCAL;
	float DHCALDeepBarrelRadius = DHCALBarrelRadius + ShiftDHCAL;

	float DHCALInnerOctRadius = DHCALBarrelRadius/cos(pi/4.0);
	float DHCALDeepOctRadius = DHCALBarrelRadius/cos(pi/4.0) + ShiftDHCAL;
	float ECALInnerOctRadius = ECALBarrelRadius/cos(pi/4.0);

	int FlagD(-1);

	if( fabs(inputPos[2]) > DHCALDeepInnerZ || fabs(inputPos[1]) > DHCALDeepBarrelRadius || fabs(inputPos[0]) > DHCALDeepBarrelRadius || fabs(inputPos[0] + inputPos[1]) > DHCALDeepOctRadius || fabs(inputPos[0] - inputPos[1]) > DHCALDeepOctRadius )
	{
		FlagD = 2;
	}
	else if( fabs(inputPos[2]) > DHCALEndCapInnerZ || fabs(inputPos[1]) > DHCALBarrelRadius || fabs(inputPos[0]) > DHCALBarrelRadius || fabs(inputPos[0] + inputPos[1]) > DHCALInnerOctRadius || fabs(inputPos[0] - inputPos[1]) > DHCALInnerOctRadius )
	{
		FlagD = 1;          // Position outsider than DHCAL Region
	}
	else if( fabs(inputPos[2]) > ECALEndCapInnerZ || fabs(inputPos[1]) > ECALBarrelRadius || fabs(inputPos[0]) > ECALBarrelRadius || fabs(inputPos[0] + inputPos[1]) > ECALInnerOctRadius || fabs(inputPos[0] - inputPos[1]) > ECALInnerOctRadius )
	{
		FlagD = 0;
	}
	else
	{
		FlagD = 10;         // Position inside Calo... Problematic for Seeds... But could be PreShower hits.
	}

	return FlagD;

}

TVector3 CalVertex( TVector3 Pos1, TVector3 Dir1, TVector3 Pos2, TVector3 Dir2 )
{
	TVector3 VertexPos;

	float tau1(0), tau2(0);

	TVector3 delP;
	delP = Pos1 - Pos2;

	double Normal(0);
	Normal = (Dir1.Dot(Dir2))*(Dir1.Dot(Dir2)) - Dir1.Mag()*Dir1.Mag()*Dir2.Mag()*Dir2.Mag();

	if(Normal != 0)
	{
		tau1 = (Dir2.Mag()*Dir2.Mag() * delP.Dot(Dir1) - Dir1.Dot(Dir2)*delP.Dot(Dir2))/Normal;
		tau2 = (Dir1.Dot(Dir2)*delP.Dot(Dir1) - Dir1.Mag()*Dir1.Mag() * delP.Dot(Dir2))/Normal;
	}

	VertexPos = 0.5*(Pos1 + Pos2 + tau1*Dir1 + tau2*Dir2);

	return VertexPos;
}

int TPCPosition( TVector3 inputPos )
{
	int flagPos(-1); // == 0 means inside TPC, == 1 means outside; 

	const float TPCRadius = 1808.0 ;    //only outer radius 
	const float TPCHalfZ = 2350.0 ;

	if( fabs(inputPos[2]) > TPCHalfZ || sqrt( inputPos[0]*inputPos[0] + inputPos[1]*inputPos[1] ) > TPCRadius ) flagPos = 1;
	else flagPos = 0;

	return flagPos;
}

float DisSeedSurface( TVector3 SeedPos )	//ECAL, HCAL, EndCapRing...
{

	//Need to treat the Overlap region specilizely...	

	float DisSS = 0;

	if( fabs(SeedPos[2]) > ECALHalfZ + 100 )
	{
		if( SeedPos.Perp() < ECALRadius + 100  )
		{
			DisSS = fabs(SeedPos[2]) - ECALHalfZ - 104;           //Depth...
		}
		else
		{
			DisSS = SeedPos.Perp() - ECALRadius; 			//Should Use Direct Distance, is it??
		}

	}
	else if( (SeedPos.Phi() > 0 && int(SeedPos.Phi() * 4/pi + 0.5) % 2 == 0 ) || (SeedPos.Phi() < 0 && int(SeedPos.Phi() * 4/pi + 8.5) % 2 == 0 ))
	{
		DisSS = min( fabs(fabs(SeedPos[0]) - ECALRadius), fabs(fabs(SeedPos[1]) - ECALRadius ) );
	}
	else
	{
		DisSS = min( fabs(fabs(SeedPos[0] + SeedPos[1])/1.414214 -ECALRadius), fabs(fabs(SeedPos[0] - SeedPos[1])/1.414214 - ECALRadius) );
	}

	return DisSS;
}

float DisTPCBoundary( TVector3 Pos )
{
	float DisZ = TMath::Min( fabs(ECALHalfZ-Pos.Z()),fabs(ECALHalfZ+Pos.Z()) );
	float DisR = ECALRadius - Pos.Perp();  
	float Dis = TMath::Min(DisZ, DisR);

	return Dis; 
}

std::vector<int>SortMeasure( std::vector<float> Measure, int ControlOrderFlag )
{

	std::vector<int> objindex;
	int Nobj = Measure.size();

	for(int k = 0; k < Nobj; k++)
	{
		objindex.push_back(k);
	}

	int FlagSwapOrder = 1;
	float SwapMeasure = 0;
	int SwapIndex = 0;

	for(int i = 0; i < Nobj && FlagSwapOrder; i++)
	{
		FlagSwapOrder = 0;
		for(int j = 0; j < Nobj - 1; j++)
		{
			if((Measure[j] < Measure[j+1] && ControlOrderFlag) || (Measure[j] > Measure[j+1] && !ControlOrderFlag) )
			{
				FlagSwapOrder = 1;
				SwapMeasure = Measure[j];
				Measure[j] = Measure[j+1];
				Measure[j+1] = SwapMeasure;

				SwapIndex = objindex[j];
				objindex[j] = objindex[j+1];
				objindex[j+1] = SwapIndex;
			}
		}
	}

	return objindex;
}

TMatrixF MatrixSummarize( TMatrixF inputMatrix )
{

	int Nrow = inputMatrix.GetNrows();
	int Ncol = inputMatrix.GetNcols();

	TMatrixF tmpMatrix(Nrow, Ncol); 

	for(int i0 = 0; i0 < Nrow; i0 ++)
	{
		for(int j0 = i0; j0 < Ncol; j0 ++)
		{
			//		if( fabs(inputMatrix(i0, j0) - 2) < 0.2 || fabs(inputMatrix(i0, j0) - 10 ) < 0.2 )	//Case 2, 3: Begin-End connector
			if(inputMatrix(i0, j0))
			{
				tmpMatrix(i0, j0) = 1;
				tmpMatrix(j0, i0) = 1;
			}
			else 
			{
				tmpMatrix(i0, j0) = 0;
				tmpMatrix(j0, i0) = 0;
			}
		}
	}

	int PreviousLinks = -1;
	int CurrentLinks = 0;
	int symloopsize = 0;
	vector <int> Indirectlinks;
	int tmpI(0);
	int tmpJ(0);

	cout<<"Matrix Type: "<<Nrow<<" * "<<Ncol<<endl;

	if( Nrow == Ncol )
	{

		while( CurrentLinks > PreviousLinks )
		{
			PreviousLinks = 0;
			CurrentLinks = 0;

			for(int i = 0; i < Nrow; i ++)
			{
				for(int j = 0; j < Ncol; j ++)
				{
					if( tmpMatrix(i, j) > 0.1 )
						PreviousLinks ++;
				}
			}

			for(int k = 0; k < Nrow; k ++)
			{
				for(int l = 0; l < Ncol; l ++)
				{
					if( tmpMatrix(k, l) > 0.1) Indirectlinks.push_back(l);
				}
				symloopsize = Indirectlinks.size();

				for(int l1(0); l1 < symloopsize; l1 ++)
				{
					tmpI = Indirectlinks[l1];
					for(int m1=l1 + 1; m1 < symloopsize; m1++)
					{
						tmpJ = Indirectlinks[m1];
						tmpMatrix(tmpI, tmpJ) = 1;
						tmpMatrix(tmpJ, tmpI) = 1;
					}
				}
				Indirectlinks.clear();
			}

			for(int u = 0; u < Nrow; u++)
			{
				for(int v = 0; v < Ncol; v++)
				{
					if( tmpMatrix(u, v) > 0.1)
						CurrentLinks ++;
				}
			}

			cout<<"Link Matrix Symmetrize Loop, PreviousFlag = "<<PreviousLinks<<", CurrentFlag = "<<CurrentLinks<<" of D = "<<Nrow<<" Matrix" <<endl;

		}
	}

	return tmpMatrix; 
}


float DistanceChargedParticleToCluster(TVector3 CPRefPos, TVector3 CPRefMom, TVector3 CluPosition)	//Extend to Track/MCP
{
	// Line extrapolation from RefPos with RefMom, calculate the minimal distance to Cluster

	float DisCPClu = 0; 
	TVector3 Diff_Clu_CPRef, NormCPRefMom; 

	Diff_Clu_CPRef = CluPosition - CPRefPos; 
	NormCPRefMom = 1.0/CPRefMom.Mag()*CPRefMom;
	float ProDis = Diff_Clu_CPRef.Dot(NormCPRefMom);	

	DisCPClu = sqrt(Diff_Clu_CPRef.Mag()*Diff_Clu_CPRef.Mag() - ProDis*ProDis);

	return DisCPClu; 
}


branchcoll ArborBranchMerge(branchcoll inputbranches, TMatrixF ConnectorMatrix)	//ABM
{
	branchcoll outputbranches; 

	int NinputBranch = inputbranches.size();
	int Nrow = ConnectorMatrix.GetNrows();
        int Ncol = ConnectorMatrix.GetNcols();
	int FlagBranchTouch[Nrow];
	int tmpCellID = 0; 

	if(Ncol != NinputBranch || Nrow != Ncol || Nrow != NinputBranch)
        {
                cout<<"Size of Connector Matrix and inputClusterColl is not match"<<endl;
        }

	for(int i0 = 0; i0 < Nrow; i0++)
	{
		FlagBranchTouch[i0] = 0;
	}

	for(int i1 = 0; i1 < Nrow; i1++)
	{
		if(FlagBranchTouch[i1] == 0)
		{
			branch Mergebranch_A = inputbranches[i1];
			branch a_MergedBranch = Mergebranch_A;
			FlagBranchTouch[i1] = 1;		

			for(int j1 = i1 + 1; j1 < Nrow; j1++)
			{
				if(FlagBranchTouch[j1] == 0 && ConnectorMatrix(i1, j1) > 0.1 )
				{
					branch Mergebranch_B = inputbranches[j1];
					FlagBranchTouch[j1] = 1;
					for(unsigned int k1 = 0; k1 < Mergebranch_B.size(); k1 ++)
					{
						tmpCellID = Mergebranch_B[k1];
						if(find(a_MergedBranch.begin(), a_MergedBranch.end(), tmpCellID) == a_MergedBranch.end())
						{
							a_MergedBranch.push_back(tmpCellID);
						}
					}					
				}
			}
			outputbranches.push_back(a_MergedBranch);
		}
	}	

	return outputbranches; 
}

