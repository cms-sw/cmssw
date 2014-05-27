#include <RecoParticleFlow/PFClusterProducer/plugins/Arbor.hh>
#include <TTree.h>
#include <algorithm>
#include <TMath.h>

using namespace std; 

std::vector<TVector3> cleanedHits;

std::vector<int> LeafHitsIndex; 
std::vector<int> JointHitsIndex; 
std::vector<int> StarJointHitsIndex; 
std::vector<int> IsoHitsIndex;
std::vector<int> SimpleSeedHitsIndex;
std::vector<int> StarSeedHitsIndex;

linkcoll Links;
linkcoll InitLinks;
linkcoll alliterlinks;
linkcoll links_debug; 
linkcoll IterLinks; 
branchcoll LengthSortBranchCollection;
branchcoll Trees; 

int NHits = 0; 
float InitLinkThreshold = 0; 
float IterLinkThreshold = 0;

void init( float CellSize, float LayerThickness ) {
	cleanedHits.clear();
	LeafHitsIndex.clear();
	JointHitsIndex.clear();
	StarJointHitsIndex.clear();
	IsoHitsIndex.clear();
	SimpleSeedHitsIndex.clear();
	StarSeedHitsIndex.clear();

	Links.clear();
	InitLinks.clear();
	IterLinks.clear();
	LengthSortBranchCollection.clear();
	Trees.clear();
	alliterlinks.clear();
	links_debug.clear();

	/*
	float Thr_1, Thr_2;		//Ecal Layer Thickness might be changed...
	Thr_1 = 2*LayerThickness; 
	Thr_2 = sqrt( LayerThickness*LayerThickness + 8 * CellSize*CellSize );

	if(Thr_1 < Thr_2) InitLinkThreshold = Thr_1;
	else InitLinkThreshold = Thr_2; 
	*/

	InitLinkThreshold = 2*LayerThickness - 0.01; 
	IterLinkThreshold = InitLinkThreshold * 2.5;

	cout<<endl<<"Thresholds"<<endl<<endl;
	cout<<"Init/Iter Threshold "<<InitLinkThreshold<<" : "<<IterLinkThreshold<<endl<<endl;

}

void HitsCleaning( std::vector<TVector3> inputHits )
{
        cleanedHits = inputHits;        //Cannot Really do much things here. Mapping before calling
        NHits = cleanedHits.size();
	/*
	for(int i = 0; i < NHits; i ++)
	{
		if(BarrelFlag(cleanedHits[i]))
			cout<<cleanedHits[i].Z()<<endl; 
	}
	*/
}

void HitsClassification( linkcoll inputLinks )
{
	int NLinks =  inputLinks.size();
	
	LeafHitsIndex.clear();
        JointHitsIndex.clear();
	StarJointHitsIndex.clear();
        IsoHitsIndex.clear();
        SimpleSeedHitsIndex.clear();
	StarSeedHitsIndex.clear();

	std::pair<int, int> a_link;

        int BeginIndex[NHits];
        int EndIndex[NHits];

        for(int i0 = 0; i0 < NHits; i0++)
        {
                BeginIndex[i0] = 0;
                EndIndex[i0] = 0;
        }

        for(int j0 = 0; j0 < NLinks; j0++)
        {
                BeginIndex[ (inputLinks[j0].first) ] ++;
                EndIndex[ (inputLinks[j0].second) ] ++;
        }

        for(int i1 = 0; i1 < NHits; i1++)
        {
                if(BeginIndex[i1] == 0 && EndIndex[i1] == 1)
                {
                        LeafHitsIndex.push_back(i1);
                }
		else if(BeginIndex[i1] == 1 && EndIndex[i1] == 1)
		{
			JointHitsIndex.push_back(i1);
		}
		else if(BeginIndex[i1] > 1 && EndIndex[i1] == 1)
		{
			StarJointHitsIndex.push_back(i1);
		}
		else if(BeginIndex[i1] == 1 && EndIndex[i1] == 0)
		{
			SimpleSeedHitsIndex.push_back(i1);
		}
		else if(BeginIndex[i1] > 1 && EndIndex[i1] == 0)
		{
			StarSeedHitsIndex.push_back(i1);
		}
		else if(BeginIndex[i1] == 0 && EndIndex[i1] == 0)
		{
			IsoHitsIndex.push_back(i1);
		}
		else
		{
			cout<<"WARNING: UNCLASSIFIED HITS, Begin Index: "<<BeginIndex[i1]<<",  End Index:  "<<EndIndex[i1]<<endl; 
		}
        }

	cout<<"Verification of Hits Classification: "<<endl;
	cout<<"Seed - Simple/Star: "<<SimpleSeedHitsIndex.size()<<" : "<<StarSeedHitsIndex.size()<<endl;
	cout<<"Joint - Simple/Star: "<<JointHitsIndex.size()<<" : "<<StarJointHitsIndex.size()<<endl;
	cout<<"Leaves: "<<LeafHitsIndex.size()<<endl;
	cout<<"IsoHits: "<<IsoHitsIndex.size()<<endl; 
	cout<<"TotalHits: "<<NHits<<endl; 
}

linkcoll LinkClean( std::vector<TVector3> allhits, linkcoll alllinks )
{
	linkcoll cleanedlinks; 

	int NLinks = alllinks.size();
        int Ncurrhitlinks = 0;
        int MinAngleIndex = -1;
        float MinAngle = 1E6;
        float tmpOrder = 0;
        float DirAngle = 0;

        std::pair<int, int> SelectedPair;

        TVector3 PosA, PosB, PosDiffAB;
	
        std::vector< std::vector<int> > LinkHits;
        LinkHits.clear();
        for(int s1 = 0; s1 < NHits; s1++)
        {
                std::vector<int> hitlink;
                for(int t1 = 0; t1 < NLinks; t1++)
                {
                        if(alllinks[t1].second == s1)
                        {
                                hitlink.push_back(alllinks[t1].first);
                        }
                }
                LinkHits.push_back(hitlink);
        }

	int count_link = 0; 

        for(int i1 = 0; i1 < NHits; i1++)
        {
                PosB = cleanedHits[i1];
                MinAngleIndex = -10;
                MinAngle = 1E6;

                std::vector<int> currhitlink = LinkHits[i1];

                Ncurrhitlinks = currhitlink.size();

                for(int k1 = 0; k1 < Ncurrhitlinks; k1++)
                {
                        PosA = cleanedHits[ currhitlink[k1] ];
                        DirAngle = (PosA + PosB).Angle(PosB - PosA);		//Replace PosA + PosB with other order parameter ~ reference direction
                        tmpOrder = (PosB - PosA).Mag() * (DirAngle + 0.1);
                        if( tmpOrder < MinAngle ) // && DirAngle < 2.5 )
                        {
                                MinAngleIndex = currhitlink[k1];
                                MinAngle = tmpOrder;
				count_link ++; 
				cout<<"AAAAAAAAAAA: "<<count_link<<endl; 
                        }
                }

                if(MinAngleIndex > -0.5)
                {
                        SelectedPair.first = MinAngleIndex;
                        SelectedPair.second = i1;
                        cleanedlinks.push_back(SelectedPair);
                }
        }

        cout<<"NStat "<<NHits<<" : "<<NLinks<<" InitLinks "<<cleanedlinks.size()<<endl;

	return cleanedlinks;
}

void BuildInitLink()
{
	Links.clear();	//all tmp links
	TVector3 PosA, PosB, PosDiffAB; 

	for(int i0 = 0; i0 < NHits; i0++)
	{
		PosA = cleanedHits[i0];
		for(int j0 = i0 + 1; j0 < NHits; j0++)
		{
			PosB = cleanedHits[j0];
			PosDiffAB = PosA - PosB;
			
			// if( PosDiffAB.Mag() < InitLinkThreshold ) // || ( PosDiffAB.Mag() < 1.6*InitLinkThreshold && PosDiffAB.Dot(PosB) < 0.9*PosDiffAB.Mag()*PosB.Mag() )  )	//Distance threshold to be optimized - should also depends on Geometry
			if( PosDiffAB.Mag() < InitLinkThreshold && PosA.Z() != PosB.Z() )
			{
				std::pair<int, int> a_Link;
				// if( PosA.Mag() > PosB.Mag() )
				if(PosA.Z() > PosB.Z() )
				{
					a_Link.first = j0;
					a_Link.second = i0; 
				}
				else
				{
					a_Link.first = i0;
					a_Link.second = j0;
				}
				Links.push_back(a_Link);
			}
		}
	}

	links_debug = Links; 
	
	cout<<"Pos NNNN "<<Links.size()<<endl; 
}

void EndPointLinkIteration()
{

	Links.clear();
	Links = InitLinks; 

	int NLeaves = LeafHitsIndex.size();
	int NStarSeeds = StarSeedHitsIndex.size();
	int NSimpleSeeds = SimpleSeedHitsIndex.size();

	TVector3 PosLeaf, PosOldSeed, Diff_Leaf_OldSeed; 
	int LeafIndex, OldSeedIndex; 
	std::pair<int, int> tmplink; 
	bool LinkOrientation; 

	//Set of reference directions...  Pos_L, Pos_S, Dir_L, Dir_S
	//Cleaning... to reduce the ambigorious
	//loop breaker?

	for(int i0 = 0; i0 < NLeaves; i0++)
	{
		LeafIndex = LeafHitsIndex[i0];
		PosLeaf = cleanedHits[ LeafIndex ];	//Depth??

		for(int j0 = 0; j0 < NSimpleSeeds + NStarSeeds; j0++)
		{
			if(j0 < NSimpleSeeds)
			{
				OldSeedIndex = SimpleSeedHitsIndex[j0];
			}
			else
			{
				OldSeedIndex = StarSeedHitsIndex[j0 - NSimpleSeeds];
			}
			PosOldSeed = cleanedHits[ OldSeedIndex ];
		
			Diff_Leaf_OldSeed = PosLeaf - PosOldSeed; 

			LinkOrientation = 0;

			if( Diff_Leaf_OldSeed.Mag() < IterLinkThreshold ) 
			{
				if( ( BarrelFlag(PosLeaf) && BarrelFlag(PosOldSeed) ) )	//HitisBarrel
				{
					if( PosLeaf.Perp()  < PosOldSeed.Perp() ) 
						LinkOrientation = 1; 						
				} 
				else if( ( !BarrelFlag(PosLeaf) && !BarrelFlag(PosOldSeed) )  )
				{
					if( fabs(PosLeaf.Z()) < fabs(PosOldSeed.Z()) )
						LinkOrientation = 1;
				}

				if( LinkOrientation && Diff_Leaf_OldSeed.Angle(PosOldSeed) < 1.0  )	//Still Need to compare with reference directions...
				{
					tmplink.first = LeafIndex;
					tmplink.second = OldSeedIndex; 
					Links.push_back(tmplink);
					cout<<"New Link added in EPLinkIteration"<<endl;
				}
			}
		}
	}

	// cout<<endl<<" NSeed: "<< NSimpleSeeds  + NStarSeeds << "  NLeaves:  "<< NLeaves <<" NInitLink "<<InitLinks.size()<<" : "<<"NCurrLink: "<<IterLinks.size()<<endl<<endl;

}

void LinkIteration()	//Energy corrections, semi-local correction
{
	IterLinks.clear();

	alliterlinks = InitLinks;
	int NInitLinks = InitLinks.size();
	TVector3 hitPos, PosA, PosB, DiffPosAB, linkDir; 
	std::pair<int, int> currlink; 

	TVector3 RefDir[NHits];

	for(int i = 0; i < NHits; i++)
	{
		hitPos = cleanedHits[i];
		RefDir[i] = 1.0/hitPos.Mag() * hitPos;
	}

	for(int j = 0; j < NInitLinks; j++)
	{
		currlink = InitLinks[j];
		PosA = cleanedHits[ currlink.first ];
		PosB = cleanedHits[ currlink.second ];
		linkDir = (PosA - PosB);		//Links are always from first point to second - verify
		linkDir *= 1.0/linkDir.Mag(); 
		RefDir[currlink.first] += 2*linkDir; 	//Weights... might be optimized...
		RefDir[currlink.second] += 4*linkDir; 
	}

	for(int i1 = 0; i1 < NHits; i1++)
	{
		RefDir[i1] *= 1.0/RefDir[i1].Mag();
		PosA = cleanedHits[i1];

		for(int j1 = i1 + 1; j1 < NHits; j1++)	
		{
			PosB = cleanedHits[j1];
			DiffPosAB = PosB - PosA; 

			if( DiffPosAB.Mag() < IterLinkThreshold && DiffPosAB.Mag() > InitLinkThreshold && DiffPosAB.Angle(RefDir[i1]) < 0.8 )	
			{

				if( PosA.Mag() > PosB.Mag() )
				{
					currlink.first = j1;
					currlink.second = i1;
				}
				else
				{
					currlink.first = i1;
					currlink.second = j1;
				}

				alliterlinks.push_back(currlink);
			} 
		}
	}

	//Reusage of link iteration codes?

	int NLinks = alliterlinks.size();
	int MinAngleIndex = -10;
	int Ncurrhitlinks = 0; 
	float MinAngle = 1E6; 
	float tmpOrder = 0;
	float DirAngle = 0; 
	std::pair<int, int> SelectedPair; 

	std::vector< std::vector<int> > LinkHits;
	LinkHits.clear();
	for(int s1 = 0; s1 < NHits; s1++)
	{
		std::vector<int> hitlink;
		for(int t1 = 0; t1 < NLinks; t1++)
		{
			if(alliterlinks[t1].second == s1)
			{
				hitlink.push_back(alliterlinks[t1].first);
			}
		}
		LinkHits.push_back(hitlink);
	}

	for(int i2 = 0; i2 < NHits; i2++)
	{
		PosB = cleanedHits[i2];
		MinAngleIndex = -10;
		MinAngle = 1E6;

		std::vector<int> currhitlink = LinkHits[i2];

		Ncurrhitlinks = currhitlink.size();

		for(int j2 = 0; j2 < Ncurrhitlinks; j2++)
		{
			PosA = cleanedHits[ currhitlink[j2] ];
			DirAngle = (RefDir[i2]).Angle(PosA - PosB);
			tmpOrder = (PosB - PosA).Mag() * (DirAngle + 1.0);
			if(tmpOrder < MinAngle) //  && DirAngle < 1.0)
			{
				MinAngleIndex = currhitlink[j2];
				MinAngle = tmpOrder;
			}
		}

		if(MinAngleIndex > -0.5)
		{
			SelectedPair.first = MinAngleIndex;
			SelectedPair.second = i2;
			IterLinks.push_back(SelectedPair);
		}
	}	

	cout<<"Init-Iter Size "<<InitLinks.size()<<" : "<<IterLinks.size()<<endl;

}

void BranchBuilding()
{
	cout<<"Build Branch"<<endl;

	int NLinks = IterLinks.size();
	int NBranches = 0;
	std::map <int, int> HitBeginIndex;
	std::map <int, int> HitEndIndex;
	std::vector< std::vector<int> > InitBranchCollection; 
	std::vector< std::vector<int> > TmpBranchCollection;
	TVector3 PosA, PosB;

	for(int i1 = 0; i1 < NHits; i1++)
	{
		HitBeginIndex[i1] = 0;
		HitEndIndex[i1] = 0;
	}

	for(int j1 = 0; j1 < NLinks; j1++)
	{
		HitBeginIndex[ (IterLinks[j1].first) ] ++;
		HitEndIndex[ (IterLinks[j1].second) ] ++;
	}

	int iterhitindex = 0; 
	int FlagInternalLoop = 0;
	for(int i2 = 0; i2 < NHits; i2++)
	{
		if(HitEndIndex[i2] > 1)
			cout<<"WARNING OF INTERNAL LOOP with more than 1 link stopped at the same Hit"<<endl;

		// cout<<"Begin/End Index "<<HitBeginIndex[i2]<<" : "<<HitEndIndex[i2]<<endl;

		if(HitBeginIndex[i2] == 0 && HitEndIndex[i2] == 1)	//EndPoint
		{
			NBranches ++; 	
			std::vector<int> currBranchhits;  	//array of indexes			
			iterhitindex = i2; 
			FlagInternalLoop = 0;
			currBranchhits.push_back(i2);		

			while(FlagInternalLoop == 0 && HitEndIndex[iterhitindex] != 0)
			{

				for(int j2 = 0; j2 < NLinks; j2++)
				{
					std::pair<int, int> PairIterator = IterLinks[j2];
					if(PairIterator.second == iterhitindex)
					{
						currBranchhits.push_back(PairIterator.first);
						iterhitindex = PairIterator.first;
						break; 
					}
				}
			}

			InitBranchCollection.push_back(currBranchhits);
		}
	}

	std::vector<float> BranchSize;
	std::vector<float> cutBranchSize;  
	std::vector<int> SortedBranchIndex; 
	std::vector<int> SortedcutBranchIndex;	
	std::vector<int> currBranch; 
	std::vector<int> iterBranch; 
	std::vector<int> touchedHits; 
	std::vector<int> leadingbranch; 
	std::vector<int> branch_A, branch_B; 

	std::map<branch, int> SortedBranchToOriginal;
	SortedBranchToOriginal.clear(); 

	int currBranchSize = 0;
	int currHit = 0;

	for(int i3 = 0; i3 < NBranches; i3++)
	{
		currBranch = InitBranchCollection[i3];
		BranchSize.push_back( float(currBranch.size()) );
	}

	SortedBranchIndex = SortMeasure(BranchSize, 1);

	for(int i4 = 0; i4 < NBranches; i4++)
	{
		currBranch = InitBranchCollection[SortedBranchIndex[i4]];

		currBranchSize = currBranch.size();

		iterBranch.clear(); 

		for(int j4 = 0; j4 < currBranchSize; j4++)
		{
			currHit = currBranch[j4];

			if( find(touchedHits.begin(), touchedHits.end(), currHit) == touchedHits.end() )
			{
				iterBranch.push_back(currHit);
				touchedHits.push_back(currHit);
			}
		}

		SortedBranchToOriginal[iterBranch] = currBranch[currBranchSize - 1];	//Map to seed...

		TmpBranchCollection.push_back(iterBranch);
		cutBranchSize.push_back( float(iterBranch.size()) );
	}

	SortedcutBranchIndex = SortMeasure(cutBranchSize, 1);

	for(int i6 = 0; i6 < NBranches; i6++)
	{
		currBranch.clear();
		currBranch = TmpBranchCollection[ SortedcutBranchIndex[i6]];
		LengthSortBranchCollection.push_back(currBranch);;
	}

	TMatrixF FlagSBMerge(NBranches, NBranches);
	int SeedIndex_A = 0; 
	int SeedIndex_B = 0; 
	TVector3 DisSeed; 

	for(int i7 = 0; i7 < NBranches; i7++)
	{
		branch_A = LengthSortBranchCollection[i7];
		SeedIndex_A = SortedBranchToOriginal[branch_A];

		for(int j7 = i7 + 1; j7 < NBranches; j7++)
		{		
			branch_B = LengthSortBranchCollection[j7];
			SeedIndex_B = SortedBranchToOriginal[branch_B];

			DisSeed = cleanedHits[ SeedIndex_A ] - cleanedHits[ SeedIndex_B ];

			if(  SeedIndex_A == SeedIndex_B )
			{
				FlagSBMerge[i7][j7] = 1.0;
				FlagSBMerge[j7][i7] = 1.0;
			}
			else if( DisSeed.Mag() < 20 )
			{
				FlagSBMerge[i7][j7] = 2.0;
				FlagSBMerge[j7][i7] = 2.0;
			}
			// else if()    Head_Tail, Small Cluster (nH < 5) Absorbing...
		}
	}

	TMatrixF SBMergeSYM = MatrixSummarize(FlagSBMerge);
	Trees = ArborBranchMerge(LengthSortBranchCollection, SBMergeSYM);

}

void BushMerging()
{
	cout<<"Merging branch"<<endl;

	int NBranch = LengthSortBranchCollection.size();
	std::vector<int> currbranch; 

	for(int i = 0; i < NBranch; i++)
	{
		currbranch = LengthSortBranchCollection[i];
	}

}

void BushAbsorbing()
{
	cout<<"Absorbing Isolated branches"<<endl;
}

void MakingCMSCluster() // edm::Event& Event, const edm::EventSetup& Setup )
{

	cout<<"Try to Make CMS Cluster"<<endl;

	int NBranches = LengthSortBranchCollection.size();
	int NHitsInBranch = 0;
	TVector3 Seed, currHit; 
	std::vector<int> currBranch; 

	for(int i0 = 0; i0 < NBranches; i0++)
	{
		currBranch = LengthSortBranchCollection[i0];
		NHitsInBranch = currBranch.size();

		cout<<i0<<" th Track has "<<currBranch.size()<<" Hits "<<endl;
		cout<<"Hits Index "<<endl; 

		for(int j0 = 0; j0 < NHitsInBranch; j0++)
		{
			cout<<currBranch[j0]<<", ";

			currHit = cleanedHits[currBranch[j0]];
		}
	}
}	

std::vector< std::vector<int> > Arbor( std::vector<TVector3> inputHits, float CellSize, float LayerThickness )
{
	init(CellSize, LayerThickness);

	HitsCleaning(inputHits);
	BuildInitLink();
	InitLinks = LinkClean( cleanedHits, Links );
	
	/*
	HitsClassification(InitLinks);
	EndPointLinkIteration();
	IterLinks = LinkClean( cleanedHits, Links );
	*/

	// LinkIteration();
	
	IterLinks = InitLinks; 

	BranchBuilding();	
	BushMerging();

	return LengthSortBranchCollection;

}
